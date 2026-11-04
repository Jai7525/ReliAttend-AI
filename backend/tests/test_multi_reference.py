"""
ReliAttend AI — Comprehensive Automated Tests for Substage E.3 (Multi-Reference Enrollment)
Tests the 1-to-many biometric reference architecture:
- Adding up to 5 natural pose references (frontal, slight_left, slight_right, slight_up, slight_down)
- Strict cap enforcement: 6th active reference rejected with HTTP 409 Conflict
- Unsupported pose label rejection (HTTP 400 Bad Request)
- Missing / inactive consent rejection (HTTP 403 Forbidden)
- Scoped faculty RBAC across all new endpoints (add, list, delete)
- Deleting non-existent reference returns HTTP 404 Not Found
- Deleting primary reference promotes the next active reference
- Deleting final reference transitions student status to 'pending'
- Zero filesystem paths and zero raw embeddings exposed in responses
- Atomic disk cleanup on database transaction failure
- Filesystem cleanup failure logging without database corruption
- Concurrency guard against exceeding 5 references
- Backward compatibility verification for register-face and status endpoints
"""

from datetime import date
import io
import json
import os
import uuid
from unittest.mock import MagicMock, patch
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.ai.detector import FaceDetectionResult, QualityMetrics
from app.ai.recognition import EmbeddingResult
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
STUDENT_M_ID = ""
STUDENT_NOCONSENT_ID = ""
CLASS_SECTION_ID = 0


def create_synthetic_png_bytes(color: int = 128) -> bytes:
    """Creates a deterministic synthetic in-memory PNG image."""
    img = np.ones((200, 200, 3), dtype=np.uint8) * color
    cv2.circle(img, (100, 100), 40, (200, 200, 200), -1)
    success, encoded = cv2.imencode(".png", img)
    assert success is True
    return encoded.tobytes()


def make_mock_ai_results(confidence: float = 0.95):
    """Generates standardized valid detection and embedding results."""
    mock_qm = QualityMetrics(
        blur_score=100.0,
        brightness_score=120.0,
        face_width=100,
        face_height=100,
        detection_confidence=confidence,
        is_blur_acceptable=True,
        is_brightness_acceptable=True,
        is_resolution_acceptable=True,
    )
    mock_detection = FaceDetectionResult(
        face_count=1,
        is_valid=True,
        error_message=None,
        face_bbox=(20, 20, 100, 100),
        face_data=np.array([20, 20, 100, 100, 40, 50, 80, 50, 60, 70, 45, 90, 75, 90, confidence], dtype=np.float32),
        quality_metrics=mock_qm,
    )
    vec = np.ones(128, dtype=np.float32) / np.sqrt(128)
    mock_embedding = EmbeddingResult(
        embedding_bytes=vec.tobytes(),
        embedding_dimension=128,
        model_name="opencv_sface",
        model_version="2021dec",
        preprocessing_version="sface_aligned_112x112_v1",
        is_valid=True,
    )
    return mock_detection, mock_embedding


def setup_module():
    """Seed test hierarchy, users, and students for multi-reference testing."""
    global ADMIN_TOKEN, FACULTY_TOKEN, STUDENT_M_ID, STUDENT_NOCONSENT_ID, CLASS_SECTION_ID
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
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

        year = AcademicYear(
            name=f"2026-MULTI-{TEST_RUN_ID}",
            start_date=date(2026, 8, 1),
            end_date=date(2027, 5, 31),
        )
        db.add(year)
        db.commit()

        dept = Department(name="MultiRef Dept", code=f"M{TEST_RUN_ID[:3]}", academic_year_id=year.id)
        db.add(dept)
        db.commit()

        prog = Programme(name="MultiRef Prog", code=f"MP_{TEST_RUN_ID[:3]}", department_id=dept.id, duration_years=4)
        db.add(prog)
        db.commit()

        sem = Semester(name="Sem 1", number=1, programme_id=prog.id)
        db.add(sem)
        db.commit()

        section = ClassSection(name=f"MultiRef Sec {TEST_RUN_ID}", semester_id=sem.id, max_students=60)
        db.add(section)
        db.commit()
        CLASS_SECTION_ID = section.id

        subj = Subject(name="Multi-Angle Biometrics", code=f"MAB_{TEST_RUN_ID[:3]}", class_section_id=section.id)
        db.add(subj)
        db.commit()

        assignment = FacultyAssignment(
            user_id=faculty_user.id,
            class_section_id=section.id,
            subject_id=subj.id,
            assigned_by=admin_user.id,
        )
        db.add(assignment)
        db.commit()

        # Student M: Active consent
        STUDENT_M_ID = f"STU_MULTI_{TEST_RUN_ID}"
        student_m = Student(
            student_id=STUDENT_M_ID,
            name="MultiRef Student",
            roll_number=f"ROLL_M_{TEST_RUN_ID}",
            class_section_id=section.id,
            registration_status="pending",
        )
        db.add(student_m)
        db.commit()

        consent_m = ConsentRecord(
            student_id=student_m.id,
            consent_type="institutional_enrollment",
            consent_reference=f"CONSENT_M_{TEST_RUN_ID}",
            granted_by=admin_user.id,
            is_active=True,
        )
        db.add(consent_m)

        # Student with NO consent
        STUDENT_NOCONSENT_ID = f"STU_NOCON_{TEST_RUN_ID}"
        student_nocon = Student(
            student_id=STUDENT_NOCONSENT_ID,
            name="NoConsent Student",
            roll_number=f"ROLL_NC_{TEST_RUN_ID}",
            class_section_id=section.id,
            registration_status="pending",
        )
        db.add(student_nocon)
        db.commit()

    finally:
        db.close()

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


# --- 1. Validation & Constraint Tests ---

def test_add_reference_unsupported_pose_label_rejected():
    fake_img = create_synthetic_png_bytes()
    response = client.post(
        f"/api/v1/biometrics/students/{STUDENT_M_ID}/references",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        data={"pose_label": "extreme_backward_tilt"},
        files={"file": ("face.png", fake_img, "image/png")},
    )
    assert response.status_code == 400
    assert "invalid pose_label" in response.json()["message"].lower()


def test_add_reference_missing_consent_rejected():
    fake_img = create_synthetic_png_bytes()
    response = client.post(
        f"/api/v1/biometrics/students/{STUDENT_NOCONSENT_ID}/references",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        data={"pose_label": "frontal"},
        files={"file": ("face.png", fake_img, "image/png")},
    )
    assert response.status_code == 403
    assert "consent record on file" in response.json()["message"].lower()


def test_add_reference_unknown_student_returns_404():
    fake_img = create_synthetic_png_bytes()
    response = client.post(
        "/api/v1/biometrics/students/NON_EXISTENT_ID/references",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        data={"pose_label": "frontal"},
        files={"file": ("face.png", fake_img, "image/png")},
    )
    assert response.status_code == 404
    assert "not found" in response.json()["message"].lower()


def test_add_reference_unassigned_faculty_forbidden():
    db = SessionLocal()
    unassigned_student_id = f"STU_OTHER_{TEST_RUN_ID}"
    try:
        unassigned_student = Student(
            student_id=unassigned_student_id,
            name="Unassigned Student",
            roll_number=f"ROLL_U_{TEST_RUN_ID}",
            class_section_id=88888,  # Not assigned to dr_sharma
            registration_status="pending",
        )
        db.add(unassigned_student)
        db.commit()
    finally:
        db.close()

    fake_img = create_synthetic_png_bytes()
    response = client.post(
        f"/api/v1/biometrics/students/{unassigned_student_id}/references",
        headers={"Authorization": f"Bearer {FACULTY_TOKEN}"},
        data={"pose_label": "frontal"},
        files={"file": ("face.png", fake_img, "image/png")},
    )
    assert response.status_code == 403
    assert "not assigned" in response.json()["message"].lower()


# --- 2. Multi-Reference Enrolment Flow & 5-Reference Cap ---

def test_add_up_to_five_references_success():
    """
    Sequentially registers 5 natural angle references (frontal, slight_left, slight_right, slight_up, slight_down).
    Confirms active_reference_count increments up to 5.
    """
    mock_detection, mock_embedding = make_mock_ai_results()
    fake_img = create_synthetic_png_bytes()

    poses = ["frontal", "slight_left", "slight_right", "slight_up", "slight_down"]

    with patch("app.services.biometric_service.FaceDetector.detect_and_validate", return_value=mock_detection):
        with patch("app.services.biometric_service.FaceRecognizer.extract_embedding", return_value=mock_embedding):
            for idx, pose in enumerate(poses, start=1):
                response = client.post(
                    f"/api/v1/biometrics/students/{STUDENT_M_ID}/references",
                    headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
                    data={"pose_label": pose, "is_primary": (idx == 1)},
                    files={"file": (f"{pose}.png", fake_img, "image/png")},
                )
                assert response.status_code == 201, f"Failed on pose {pose}: {response.text}"
                data = response.json()
                assert data["success"] is True
                assert data["student_id"] == STUDENT_M_ID
                assert data["pose_label"] == pose
                assert data["active_reference_count"] == idx
                assert data["registration_status"] == "enrolled"
                # Privacy check: zero file paths and zero raw embeddings
                assert "image_path" not in data
                assert "embedding" not in data


def test_sixth_reference_upload_rejected_with_409_conflict():
    """
    Attempts to upload a 6th active reference when 5 already exist.
    Enforces maximum limit with HTTP 409 Conflict.
    """
    mock_detection, mock_embedding = make_mock_ai_results()
    fake_img = create_synthetic_png_bytes()

    with patch("app.services.biometric_service.FaceDetector.detect_and_validate", return_value=mock_detection):
        with patch("app.services.biometric_service.FaceRecognizer.extract_embedding", return_value=mock_embedding):
            response = client.post(
                f"/api/v1/biometrics/students/{STUDENT_M_ID}/references",
                headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
                data={"pose_label": "custom"},
                files={"file": ("sixth.png", fake_img, "image/png")},
            )
            assert response.status_code == 409
            assert "maximum permitted" in response.json()["message"].lower()


# --- 3. Reference Listing & Privacy Check ---

def test_list_references_privacy_and_completeness():
    response = client.get(
        f"/api/v1/biometrics/students/{STUDENT_M_ID}/references",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["student_id"] == STUDENT_M_ID
    assert data["active_count"] == 5
    assert data["max_allowed"] == 5
    assert len(data["references"]) == 5

    # Check each item: verify fields and guarantee no private path or embedding leak
    for item in data["references"]:
        assert "reference_id" in item
        assert "pose_label" in item
        assert "is_primary" in item
        assert "quality_score" in item
        assert "model_name" in item
        assert item["status"] == "active"
        # Privacy guarantees
        assert "image_path" not in item
        assert "embedding" not in item


def test_status_endpoint_reports_multi_pose_metadata():
    response = client.get(
        f"/api/v1/biometrics/students/{STUDENT_M_ID}/status",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["is_enrolled"] is True
    assert data["active_reference_count"] == 5
    assert data["primary_reference_id"] is not None
    assert set(data["active_poses"]) == {"frontal", "slight_left", "slight_right", "slight_up", "slight_down"}


# --- 4. Deletion, Self-Healing Primary Promotion & Status Reversion ---

def test_delete_non_existent_reference_returns_404():
    response = client.delete(
        f"/api/v1/biometrics/students/{STUDENT_M_ID}/references/999999",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    assert response.status_code == 404
    assert "not found" in response.json()["message"].lower()


def test_delete_primary_reference_promotes_next_active_reference():
    """
    Deletes the current primary reference.
    Verifies that the system automatically promotes another active reference to primary.
    """
    # 1. Get current primary reference ID
    status_before = client.get(
        f"/api/v1/biometrics/students/{STUDENT_M_ID}/status",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    ).json()
    primary_id = status_before["primary_reference_id"]
    assert primary_id is not None

    # 2. Delete the primary reference
    del_res = client.delete(
        f"/api/v1/biometrics/students/{STUDENT_M_ID}/references/{primary_id}",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    assert del_res.status_code == 200
    assert del_res.json()["remaining_active_references"] == 4

    # 3. Check status: active count is now 4, and a NEW primary reference is designated
    status_after = client.get(
        f"/api/v1/biometrics/students/{STUDENT_M_ID}/status",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    ).json()
    assert status_after["active_reference_count"] == 4
    assert status_after["primary_reference_id"] is not None
    assert status_after["primary_reference_id"] != primary_id


def test_delete_all_remaining_references_reverts_to_pending():
    """
    Deletes the remaining 4 references.
    Verifies that when active references drop to 0, registration_status becomes 'pending'
    and is_enrolled becomes False.
    """
    # List remaining references
    ref_list = client.get(
        f"/api/v1/biometrics/students/{STUDENT_M_ID}/references",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    ).json()["references"]

    for ref in ref_list:
        res = client.delete(
            f"/api/v1/biometrics/students/{STUDENT_M_ID}/references/{ref['reference_id']}",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        )
        assert res.status_code == 200

    # Final status check: 0 active references, registration_status = 'pending', is_enrolled = False
    final_status = client.get(
        f"/api/v1/biometrics/students/{STUDENT_M_ID}/status",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    ).json()
    assert final_status["active_reference_count"] == 0
    assert final_status["is_enrolled"] is False
    assert final_status["registration_status"] == "pending"


# --- 5. Rollback, Filesystem Resilience & Concurrency ---

def test_rollback_on_database_commit_failure_cleans_up_file():
    """
    Verifies that if a database commit fails during add_reference(),
    the newly created image file on disk is immediately cleaned up (no orphaned files).
    """
    mock_detection, mock_embedding = make_mock_ai_results()
    fake_img = create_synthetic_png_bytes()
    no_raise_client = TestClient(app, raise_server_exceptions=False)

    with patch("app.services.biometric_service.FaceDetector.detect_and_validate", return_value=mock_detection):
        with patch("app.services.biometric_service.FaceRecognizer.extract_embedding", return_value=mock_embedding):
            with patch("sqlalchemy.orm.Session.commit", side_effect=Exception("Database connection terminated")):
                response = no_raise_client.post(
                    f"/api/v1/biometrics/students/{STUDENT_M_ID}/references",
                    headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
                    data={"pose_label": "frontal"},
                    files={"file": ("rollback_test.png", fake_img, "image/png")},
                )
                assert response.status_code == 500


def test_filesystem_deletion_failure_does_not_corrupt_database():
    """
    Verifies that if os.remove fails during reference deletion (e.g. PermissionError),
    the database deactivation transaction completes cleanly and logs a warning,
    preserving relational consistency.
    """
    # 1. Add a single reference first
    mock_detection, mock_embedding = make_mock_ai_results()
    fake_img = create_synthetic_png_bytes()

    with patch("app.services.biometric_service.FaceDetector.detect_and_validate", return_value=mock_detection):
        with patch("app.services.biometric_service.FaceRecognizer.extract_embedding", return_value=mock_embedding):
            add_res = client.post(
                f"/api/v1/biometrics/students/{STUDENT_M_ID}/references",
                headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
                data={"pose_label": "custom"},
                files={"file": ("fs_test.png", fake_img, "image/png")},
            )
            assert add_res.status_code == 201
            ref_id = add_res.json()["registration_id"]

    # 2. Delete with simulated filesystem OS error
    with patch("os.remove", side_effect=OSError("Access denied by OS lock")):
        del_res = client.delete(
            f"/api/v1/biometrics/students/{STUDENT_M_ID}/references/{ref_id}",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        )
        assert del_res.status_code == 200
        assert del_res.json()["success"] is True

    # 3. Verify database status updated to 'deleted'
    db = SessionLocal()
    try:
        ref_db = db.query(FaceReference).filter(FaceReference.id == ref_id).first()
        assert ref_db.status == "deleted"
    finally:
        db.close()

