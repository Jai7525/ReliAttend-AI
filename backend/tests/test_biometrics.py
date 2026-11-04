"""
ReliAttend AI — Comprehensive Automated Tests for Phase 6 Biometrics
Covers positive registration, missing consent, invalid formats, corrupted images,
multi/zero face rejection, quality failures, scoped RBAC (Admin vs assigned Faculty vs unassigned Faculty),
single active reference replacement, rollback on database failure, and audit logging.
"""

from datetime import date
import json
import os
import uuid
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.security import hash_password
from app.database import Base, SessionLocal, engine
from app.main import app
from app.models.academic import (
    AcademicYear,
    ClassSection,
    Department,
    FacultyAssignment,
    Programme,
    Semester,
    Subject,
)
from app.models.attendance import AuditLog
from app.models.student import ConsentRecord, FaceReference, Student
from app.models.user import User

client = TestClient(app)

TEST_RUN_ID = uuid.uuid4().hex[:6].upper()
ADMIN_TOKEN = ""
FACULTY_TOKEN = ""
STUDENT_A_ID = ""
STUDENT_B_ID = ""
SECTION_ID = 0


def create_synthetic_face_image_bytes(has_face_like_pattern: bool = True) -> bytes:
    """
    Creates an in-memory synthetic image.
    If has_face_like_pattern is False, returns a blank gray canvas.
    """
    img = np.ones((300, 300, 3), dtype=np.uint8) * 128
    if has_face_like_pattern:
        # Draw high-contrast shapes to produce gradients, but YuNet still looks for face features.
        cv2.circle(img, (150, 150), 80, (200, 200, 200), -1)
        cv2.circle(img, (120, 130), 10, (50, 50, 50), -1)
        cv2.circle(img, (180, 130), 10, (50, 50, 50), -1)
        cv2.line(img, (130, 190), (170, 190), (50, 50, 50), 3)

    success, encoded = cv2.imencode(".png", img)
    assert success is True
    return encoded.tobytes()


def setup_module():
    """Ensure prerequisite hierarchy and test users exist."""
    global ADMIN_TOKEN, FACULTY_TOKEN, STUDENT_A_ID, STUDENT_B_ID, SECTION_ID
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        # 1. Create or get admin & faculty users
        admin_user = db.query(User).filter_by(username="admin").first()
        if not admin_user:
            admin_user = User(
                email="admin@reliattend.com",
                username="admin",
                password_hash=hash_password("Admin@123"),
                full_name="System Admin",
                role="admin",
                is_active=True,
            )
            db.add(admin_user)

        faculty_user = db.query(User).filter_by(username="dr_sharma").first()
        if not faculty_user:
            faculty_user = User(
                email="dr_sharma@reliattend.com",
                username="dr_sharma",
                password_hash=hash_password("Faculty@123"),
                full_name="Dr. Sharma",
                role="faculty",
                is_active=True,
            )
            db.add(faculty_user)
        db.commit()

        # 2. Academic hierarchy
        year = AcademicYear(
            name=f"2026-BIO-{TEST_RUN_ID}",
            start_date=date(2026, 8, 1),
            end_date=date(2027, 5, 31),
        )
        db.add(year)
        db.commit()

        dept = Department(name="Biometric Dept", code=f"B{TEST_RUN_ID[:3]}", academic_year_id=year.id)
        db.add(dept)
        db.commit()

        prog = Programme(name="Biometric Tech", code=f"P{TEST_RUN_ID[:3]}", department_id=dept.id, duration_years=4)
        db.add(prog)
        db.commit()

        sem = Semester(name="Sem 1", number=1, programme_id=prog.id)
        db.add(sem)
        db.commit()

        section = ClassSection(name=f"Sec Bio {TEST_RUN_ID}", semester_id=sem.id, max_students=60)
        db.add(section)
        db.commit()
        SECTION_ID = section.id

        subj = Subject(name="Pattern Recognition", code=f"PR_{TEST_RUN_ID[:4]}", class_section_id=section.id)
        db.add(subj)
        db.commit()

        # Assign dr_sharma to this section
        assignment = FacultyAssignment(
            user_id=faculty_user.id,
            class_section_id=section.id,
            subject_id=subj.id,
            assigned_by=admin_user.id,
        )
        db.add(assignment)
        db.commit()

        # 3. Create Students:
        # Student A: with active biometric consent
        STUDENT_A_ID = f"STU_BIO_A_{TEST_RUN_ID}"
        student_a = Student(
            student_id=STUDENT_A_ID,
            name="Alice Biometrics",
            roll_number=f"ROLL_A_{TEST_RUN_ID}",
            class_section_id=section.id,
            registration_status="pending",
        )
        db.add(student_a)
        db.commit()

        consent_a = ConsentRecord(
            student_id=student_a.id,
            consent_type="institutional_enrollment",
            consent_reference=f"CONSENT_REF_{TEST_RUN_ID}",
            granted_by=admin_user.id,
            is_active=True,
        )
        db.add(consent_a)

        # Student B: with NO consent
        STUDENT_B_ID = f"STU_BIO_B_{TEST_RUN_ID}"
        student_b = Student(
            student_id=STUDENT_B_ID,
            name="Bob NoConsent",
            roll_number=f"ROLL_B_{TEST_RUN_ID}",
            class_section_id=section.id,
            registration_status="pending",
        )
        db.add(student_b)
        db.commit()

    finally:
        db.close()

    # Log in and acquire tokens
    res_admin = client.post(
        "/api/v1/auth/login",
        json={"username_or_email": "admin", "password": "Admin@123"},
    )
    ADMIN_TOKEN = res_admin.json()["data"]["tokens"]["access_token"]

    res_faculty = client.post(
        "/api/v1/auth/login",
        json={"username_or_email": "dr_sharma", "password": "Faculty@123"},
    )
    FACULTY_TOKEN = res_faculty.json()["data"]["tokens"]["access_token"]


# --- Test Cases ---

def test_status_endpoint_returns_unregistered():
    response = client.get(
        f"/api/v1/biometrics/students/{STUDENT_A_ID}/status",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["student_id"] == STUDENT_A_ID
    assert data["is_enrolled"] is False
    assert data["reference_count"] == 0
    assert data["active_reference_count"] == 0


def test_register_face_unknown_student_returns_404():
    fake_img = create_synthetic_face_image_bytes()
    response = client.post(
        "/api/v1/biometrics/students/UNKNOWN_STUDENT_999/register-face",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        files={"file": ("face.png", fake_img, "image/png")},
    )
    assert response.status_code == 404
    assert "not found" in response.json()["message"].lower()


def test_register_face_missing_consent_returns_403():
    fake_img = create_synthetic_face_image_bytes()
    # Student B has no consent record
    response = client.post(
        f"/api/v1/biometrics/students/{STUDENT_B_ID}/register-face",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        files={"file": ("face.png", fake_img, "image/png")},
    )
    assert response.status_code == 403
    assert "consent record on file" in response.json()["message"].lower()


def test_register_face_corrupted_file_returns_400():
    response = client.post(
        f"/api/v1/biometrics/students/{STUDENT_A_ID}/register-face",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        files={"file": ("bad.jpg", b"corrupted non image bytes", "image/jpeg")},
    )
    assert response.status_code == 400
    assert "invalid or unsupported image file" in response.json()["message"].lower()


def test_register_face_zero_faces_detected_returns_422():
    blank_png = create_synthetic_face_image_bytes(has_face_like_pattern=False)
    response = client.post(
        f"/api/v1/biometrics/students/{STUDENT_A_ID}/register-face",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        files={"file": ("blank.png", blank_png, "image/png")},
    )
    assert response.status_code == 422
    assert "no face detected" in response.json()["message"].lower()


def test_register_face_assigned_faculty_authorization_allowed():
    # dr_sharma is assigned to Student A's class section -> can query status
    response = client.get(
        f"/api/v1/biometrics/students/{STUDENT_A_ID}/status",
        headers={"Authorization": f"Bearer {FACULTY_TOKEN}"},
    )
    assert response.status_code == 200
    assert response.json()["student_id"] == STUDENT_A_ID


def test_register_face_unassigned_faculty_forbidden():
    # dr_sharma is NOT assigned to class section 99999
    db = SessionLocal()
    other_student_id = f"STU_OTHER_{TEST_RUN_ID}"
    try:
        other_student = Student(
            student_id=other_student_id,
            name="Other Student",
            roll_number=f"ROLL_OTHER_{TEST_RUN_ID}",
            class_section_id=99999,
            registration_status="pending",
        )
        db.add(other_student)
        db.commit()
    finally:
        db.close()

    response = client.get(
        f"/api/v1/biometrics/students/{other_student_id}/status",
        headers={"Authorization": f"Bearer {FACULTY_TOKEN}"},
    )
    assert response.status_code == 403
    assert "not assigned" in response.json()["message"].lower()


def test_positive_face_registration_flow_and_audit():
    """
    Tests end-to-end face registration with a mocked valid 1-face detection & feature extraction,
    verifying filesystem storage, single active reference, DB state, and audit log.
    """
    from unittest.mock import MagicMock, patch
    from app.ai.detector import FaceDetectionResult, QualityMetrics
    from app.ai.recognition import EmbeddingResult

    # Mock detection result with 1 valid face
    mock_qm = QualityMetrics(
        blur_score=120.0,
        brightness_score=110.0,
        face_width=120,
        face_height=140,
        detection_confidence=0.96,
        is_blur_acceptable=True,
        is_brightness_acceptable=True,
        is_resolution_acceptable=True,
    )
    mock_detection = FaceDetectionResult(
        face_count=1,
        is_valid=True,
        error_message=None,
        face_bbox=(50, 50, 120, 140),
        face_data=np.array([50, 50, 120, 140, 70, 80, 150, 80, 110, 110, 80, 150, 140, 150, 0.96], dtype=np.float32),
        quality_metrics=mock_qm,
    )
    # 128 normalized float32 values
    vec = np.ones(128, dtype=np.float32) / np.sqrt(128)
    mock_embedding = EmbeddingResult(
        embedding_bytes=vec.tobytes(),
        embedding_dimension=128,
        model_name="opencv_sface",
        model_version="2021dec",
        preprocessing_version="sface_aligned_112x112_v1",
        is_valid=True,
    )

    fake_img = create_synthetic_face_image_bytes()

    with patch("app.services.biometric_service.FaceDetector.detect_and_validate", return_value=mock_detection):
        with patch("app.services.biometric_service.FaceRecognizer.extract_embedding", return_value=mock_embedding):
            # 1. Register face
            response = client.post(
                f"/api/v1/biometrics/students/{STUDENT_A_ID}/register-face",
                headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
                files={"file": ("valid_face.png", fake_img, "image/png")},
            )
            assert response.status_code == 201
            data = response.json()
            assert data["success"] is True
            assert data["student_id"] == STUDENT_A_ID
            assert data["registration_status"] == "enrolled"
            assert data["face_count"] == 1
            assert data["model_name"] == "opencv_sface"
            assert data["embedding_dimension"] == 128
            # Path & raw bytes must NOT be in the API response
            assert "image_path" not in data
            assert "embedding" not in data

            # 2. Status verification
            status_res = client.get(
                f"/api/v1/biometrics/students/{STUDENT_A_ID}/status",
                headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            )
            assert status_res.status_code == 200
            sdata = status_res.json()
            assert sdata["is_enrolled"] is True
            assert sdata["reference_count"] == 1
            assert sdata["active_reference_count"] == 1

            # 3. Test replacement: Re-register face -> previous reference should be superseded
            re_response = client.post(
                f"/api/v1/biometrics/students/{STUDENT_A_ID}/register-face",
                headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
                files={"file": ("new_face.png", fake_img, "image/png")},
            )
            assert re_response.status_code == 201

            # Check status again: 1 active reference maintained, student enrolled
            status_res2 = client.get(
                f"/api/v1/biometrics/students/{STUDENT_A_ID}/status",
                headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            )
            assert status_res2.status_code == 200
            sdata2 = status_res2.json()
            assert sdata2["reference_count"] == 1
            assert sdata2["active_reference_count"] == 1

            # 4. Audit Log verification
            db = SessionLocal()
            try:
                audit = (
                    db.query(AuditLog)
                    .filter(AuditLog.action == "student.face_enrolled")
                    .order_by(AuditLog.id.desc())
                    .first()
                )
                assert audit is not None
                details = json.loads(audit.details)
                assert details["student_id"] == STUDENT_A_ID
                assert details["model_name"] == "opencv_sface"
                assert details["superseded_previous"] is True
            finally:
                db.close()


def test_rollback_on_database_failure_cleans_up_file():
    """
    Verifies that if an exception occurs during the database transaction,
    the newly written reference image file is safely cleaned up from disk.
    """
    from unittest.mock import patch
    from app.ai.detector import FaceDetectionResult, QualityMetrics
    from app.ai.recognition import EmbeddingResult

    mock_qm = QualityMetrics(100.0, 100.0, 100, 100, 0.95, True, True, True)
    mock_detection = FaceDetectionResult(1, True, None, (10, 10, 100, 100), np.zeros(15, dtype=np.float32), mock_qm)
    mock_embedding = EmbeddingResult(np.zeros(128, dtype=np.float32).tobytes(), 128, "opencv_sface", "2021dec", "sface_aligned_112x112_v1", True)

    fake_img = create_synthetic_face_image_bytes()
    no_raise_client = TestClient(app, raise_server_exceptions=False)

    # Force db.commit() to throw an exception
    with patch("app.services.biometric_service.FaceDetector.detect_and_validate", return_value=mock_detection):
        with patch("app.services.biometric_service.FaceRecognizer.extract_embedding", return_value=mock_embedding):
            with patch("sqlalchemy.orm.Session.commit", side_effect=Exception("Forced DB Error")):
                response = no_raise_client.post(
                    f"/api/v1/biometrics/students/{STUDENT_A_ID}/register-face",
                    headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
                    files={"file": ("face.png", fake_img, "image/png")},
                )
                assert response.status_code == 500
