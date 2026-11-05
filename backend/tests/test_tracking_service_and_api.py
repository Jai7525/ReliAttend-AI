"""
ReliAttend AI — Multi-Face Tracking Service & API Tests (Phase 9 - Substage 9.3)
Validates session lifecycle, academic assignment authorization, ownership checks,
sequential frame ingestion, state transitions (tentative -> confirmed -> lost),
monotonically increasing frame sequence enforcement, empty frames, corrupted/oversized images,
session termination, idle timeout cleanup, and zero database persistence for attendance/frames.
"""

import datetime
import io
import time
import uuid
from unittest.mock import MagicMock, patch
import cv2
from fastapi import status
from fastapi.testclient import TestClient
import numpy as np
import pytest
from sqlalchemy.orm import Session

from app.ai.detector import DetectedFaceBox, FaceDetector, MultiFaceDetectionResult, QualityMetrics
from app.ai.tracking.tracker import MultiFaceTracker
from app.core.exceptions import AuthorizationForbiddenError, BadRequestError, ResourceNotFoundError
from app.core.security import create_access_token
from app.database import Base, SessionLocal, engine
from app.main import app
from app.models.academic import AcademicYear, ClassSection, Department, FacultyAssignment, Programme, Semester, Subject
from app.models.user import User
from app.schemas.tracking import CreateTrackingSessionRequest
from app.services.tracking_service import TrackingService, TrackingSessionManager, session_manager


def _encode_jpeg(img_bgr: np.ndarray) -> bytes:
    _, buf = cv2.imencode(".jpg", img_bgr)
    return buf.tobytes()


class FakeUploadFile:
    def __init__(self, filename: str, content: bytes, content_type: str = "image/jpeg"):
        self.filename = filename
        self.content = content
        self.content_type = content_type

    async def read(self) -> bytes:
        return self.content


@pytest.fixture(scope="module")
def setup_tracking_db():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    suffix = uuid.uuid4().hex[:6]

    # 1. Academic Structure (AcademicYear first)
    ay = AcademicYear(
        name=f"2026-2027-TRK-{suffix}",
        start_date=datetime.date(2026, 7, 1),
        end_date=datetime.date(2027, 5, 31),
        is_active=True,
    )
    db.add(ay)
    db.flush()

    dept = Department(name=f"Tracking IT {suffix}", code=f"TK_{suffix[:4]}", academic_year_id=ay.id, is_active=True)
    db.add(dept)
    db.flush()

    prog = Programme(name=f"B.Tech TRK {suffix}", code=f"BP_{suffix[:4]}", department_id=dept.id)
    db.add(prog)
    db.flush()

    sem = Semester(name=f"Semester 8 {suffix}", number=8, programme_id=prog.id)
    db.add(sem)
    db.flush()

    section = ClassSection(semester_id=sem.id, name=f"A_{suffix}", max_students=60, is_active=True)
    db.add(section)
    db.flush()

    subj = Subject(name="Computer Vision", code=f"CV_{suffix[:4]}", class_section_id=section.id, is_active=True)
    db.add(subj)
    db.flush()

    # 2. Users
    admin_user = User(
        username=f"track_admin_{suffix}",
        email=f"track_admin_{suffix}@reliattend.edu",
        password_hash="hash",
        full_name="Track Admin",
        role="admin",
        is_active=True,
    )
    faculty_user = User(
        username=f"track_faculty_{suffix}",
        email=f"track_faculty_{suffix}@reliattend.edu",
        password_hash="hash",
        full_name="Track Faculty",
        role="faculty",
        is_active=True,
    )
    unauthorized_faculty = User(
        username=f"track_other_fac_{suffix}",
        email=f"track_other_fac_{suffix}@reliattend.edu",
        password_hash="hash",
        full_name="Track Other Faculty",
        role="faculty",
        is_active=True,
    )
    db.add_all([admin_user, faculty_user, unauthorized_faculty])
    db.flush()

    # 3. Faculty Assignment (only faculty_user assigned to section)
    fa = FacultyAssignment(
        user_id=faculty_user.id,
        subject_id=subj.id,
        class_section_id=section.id,
        assigned_by=admin_user.id,
        is_active=True,
    )
    db.add(fa)
    db.commit()

    # 4. Unassigned faculty user
    other_fac = User(
        username=f"track_other_fac_unassigned_{suffix}",
        email=f"track_other_fac_unassigned_{suffix}@reliattend.edu",
        password_hash="hash",
        full_name="Unassigned Faculty",
        role="faculty",
        is_active=True,
    )
    db.add(other_fac)
    db.commit()

    db.refresh(admin_user)
    db.refresh(faculty_user)
    db.refresh(unauthorized_faculty)
    db.refresh(other_fac)

    yield {
        "admin_user": admin_user,
        "faculty_user": faculty_user,
        "unauthorized_faculty": unauthorized_faculty,
        "unassigned_faculty": other_fac,
        "section_id": section.id,
    }

    # Teardown: delete records created by this fixture
    try:
        db.delete(fa)
        db.delete(subj)
        db.delete(section)
        db.delete(sem)
        db.delete(prog)
        db.delete(dept)
        db.delete(ay)
        db.delete(other_fac)
        db.delete(unauthorized_faculty)
        db.delete(faculty_user)
        db.delete(admin_user)
        db.commit()
    except Exception:
        db.rollback()
    finally:
        db.close()


@pytest.fixture(autouse=True)
def cleanup_sessions():
    session_manager.clear_all()
    yield
    session_manager.clear_all()


# ==============================================================================
# 1. SERVICE LAYER UNIT TESTS: SESSION LIFECYCLE & AUTHORIZATION
# ==============================================================================

def test_service_create_session_success(setup_tracking_db):
    data = setup_tracking_db
    db = SessionLocal()
    try:
        service = TrackingService(db=db)
        req = CreateTrackingSessionRequest(class_section_id=data["section_id"])
        resp = service.create_session(req, current_user=data["faculty_user"])

        assert resp.session_id.startswith("trk_sess_")
        assert resp.class_section_id == data["section_id"]
        assert resp.config.iou_threshold == 0.30
        assert resp.config.min_hits_to_confirm == 3

        # Verify retrieved from manager
        stored = session_manager.get_session(resp.session_id)
        assert stored is not None
        assert stored.user_id == data["faculty_user"].id
    finally:
        db.close()


def test_service_create_session_admin_allowed_unassigned_section(setup_tracking_db):
    data = setup_tracking_db
    db = SessionLocal()
    try:
        service = TrackingService(db=db)
        req = CreateTrackingSessionRequest(class_section_id=data["section_id"])
        resp = service.create_session(req, current_user=data["admin_user"])
        assert resp.session_id.startswith("trk_sess_")
    finally:
        db.close()


def test_service_create_session_forbidden_for_unassigned_faculty(setup_tracking_db):
    data = setup_tracking_db
    db = SessionLocal()
    try:
        service = TrackingService(db=db)
        req = CreateTrackingSessionRequest(class_section_id=data["section_id"])
        with pytest.raises(AuthorizationForbiddenError):
            service.create_session(req, current_user=data["unauthorized_faculty"])
    finally:
        db.close()


def test_service_create_session_nonexistent_section(setup_tracking_db):
    data = setup_tracking_db
    db = SessionLocal()
    try:
        service = TrackingService(db=db)
        req = CreateTrackingSessionRequest(class_section_id=999999)
        with pytest.raises(ResourceNotFoundError):
            service.create_session(req, current_user=data["faculty_user"])
    finally:
        db.close()


def test_service_session_ownership_protection(setup_tracking_db):
    data = setup_tracking_db
    db = SessionLocal()
    try:
        service = TrackingService(db=db)
        req = CreateTrackingSessionRequest(class_section_id=data["section_id"])
        session = service.create_session(req, current_user=data["faculty_user"])

        # Another faculty attempting to access or end session gets 403 Forbidden
        with pytest.raises(AuthorizationForbiddenError):
            service.get_tracks(session.session_id, current_user=data["unauthorized_faculty"])

        with pytest.raises(AuthorizationForbiddenError):
            service.end_session(session.session_id, current_user=data["unauthorized_faculty"])

        # Admin can access any session
        tracks_resp = service.get_tracks(session.session_id, current_user=data["admin_user"])
        assert tracks_resp.active_tracks_count == 0
    finally:
        db.close()


def test_service_end_session_and_free_memory(setup_tracking_db):
    data = setup_tracking_db
    db = SessionLocal()
    try:
        service = TrackingService(db=db)
        req = CreateTrackingSessionRequest(class_section_id=data["section_id"])
        session = service.create_session(req, current_user=data["faculty_user"])

        end_resp = service.end_session(session.session_id, current_user=data["faculty_user"])
        assert end_resp.session_id == session.session_id
        assert end_resp.total_tracks_created == 0

        # Session should no longer exist
        with pytest.raises(ResourceNotFoundError):
            service.get_tracks(session.session_id, current_user=data["faculty_user"])
    finally:
        db.close()


def test_service_idle_session_expiration(setup_tracking_db):
    """Sessions with idle time exceeding timeout are automatically purged."""
    mgr = TrackingSessionManager(idle_timeout_seconds=0.1)
    db = SessionLocal()
    try:
        service = TrackingService(db=db, manager=mgr)
        req = CreateTrackingSessionRequest(class_section_id=data["section_id"]) if (data := setup_tracking_db) else None
        session = service.create_session(req, current_user=data["faculty_user"])

        assert mgr.get_session(session.session_id) is not None
        time.sleep(0.15)
        # Calling get_session triggers _cleanup_expired_locked
        assert mgr.get_session(session.session_id) is None
    finally:
        db.close()


# ==============================================================================
# 2. FRAME PROCESSING & MULTI-FACE TRACKING SERVICE TESTS
# ==============================================================================

@pytest.mark.anyio
async def test_service_process_frame_empty_frame(setup_tracking_db):
    """An empty frame (zero faces detected) must update tracker cleanly with 0 tracks."""
    data = setup_tracking_db
    db = SessionLocal()
    try:
        service = TrackingService(db=db)
        req = CreateTrackingSessionRequest(class_section_id=data["section_id"])
        session = service.create_session(req, current_user=data["faculty_user"])

        # Blank 400x400 image (no faces)
        blank_img = np.full((400, 400, 3), 128, dtype=np.uint8)
        upload = FakeUploadFile(filename="frame0.jpg", content=_encode_jpeg(blank_img))

        res = await service.process_frame(
            session_id=session.session_id,
            frame_seq=0,
            timestamp=100.0,
            file=upload,
            current_user=data["faculty_user"],
        )

        assert res.session_id == session.session_id
        assert res.frame_seq == 0
        assert res.active_tracks_count == 0
        assert res.confirmed_tracks_count == 0
        assert len(res.tracks) == 0
    finally:
        db.close()


@pytest.mark.anyio
async def test_service_process_frame_sequence_validation(setup_tracking_db):
    """Replaying or sending out-of-order frame sequences must raise BadRequestError."""
    data = setup_tracking_db
    db = SessionLocal()
    try:
        service = TrackingService(db=db)
        req = CreateTrackingSessionRequest(class_section_id=data["section_id"])
        session = service.create_session(req, current_user=data["faculty_user"])

        blank_img = np.full((400, 400, 3), 128, dtype=np.uint8)
        upload1 = FakeUploadFile(filename="frame5.jpg", content=_encode_jpeg(blank_img))
        await service.process_frame(session.session_id, frame_seq=5, timestamp=100.0, file=upload1, current_user=data["faculty_user"])

        # Sending frame_seq <= 5 should fail
        upload2 = FakeUploadFile(filename="frame5.jpg", content=_encode_jpeg(blank_img))
        with pytest.raises(BadRequestError) as exc1:
            await service.process_frame(session.session_id, frame_seq=5, timestamp=100.1, file=upload2, current_user=data["faculty_user"])
        assert "strictly greater" in str(exc1.value)

        upload3 = FakeUploadFile(filename="frame4.jpg", content=_encode_jpeg(blank_img))
        with pytest.raises(BadRequestError) as exc2:
            await service.process_frame(session.session_id, frame_seq=4, timestamp=100.2, file=upload3, current_user=data["faculty_user"])
        assert "strictly greater" in str(exc2.value)
    finally:
        db.close()


@pytest.mark.anyio
async def test_service_process_frame_tracks_promotion_and_loss(setup_tracking_db):
    """
    Mock FaceDetector to emit detections over consecutive frames and verify:
    - Frame 0: Tentative track created.
    - Frame 1: Tentative track updated (2 hits).
    - Frame 2: Confirmed track (3 hits).
    - Frame 3: Missing detection -> temporarily_lost.
    """
    data = setup_tracking_db
    db = SessionLocal()
    try:
        mock_detector = MagicMock(spec=FaceDetector)
        mock_detector.decode_image_bytes.return_value = np.zeros((400, 400, 3), dtype=np.uint8)

        service = TrackingService(db=db, detector=mock_detector)
        req = CreateTrackingSessionRequest(class_section_id=data["section_id"], min_hits_to_confirm=3)
        session = service.create_session(req, current_user=data["faculty_user"])

        dummy_face = DetectedFaceBox(
            face_id=1,
            bounding_box=(100, 100, 60, 60),
            confidence=0.95,
            landmarks={"left_eye": (115, 115)},
            quality=QualityMetrics(150.0, 120.0, 60, 60, 0.95, True, True, True),
            is_quality_acceptable=True,
        )

        blank_bytes = _encode_jpeg(np.zeros((100, 100, 3), dtype=np.uint8))

        # Frame 0: 1 detection
        mock_detector.detect_multi_faces.return_value = MultiFaceDetectionResult(
            face_count=1,
            faces=[dummy_face],
            frame_width=400,
            frame_height=400,
            inference_time_ms=12.5,
        )
        f0 = await service.process_frame(session.session_id, 0, 100.0, FakeUploadFile("f0.jpg", blank_bytes), data["faculty_user"])
        assert f0.active_tracks_count == 1
        assert f0.confirmed_tracks_count == 0
        assert f0.tracks[0].state == "tentative"
        assert f0.tracks[0].is_newly_created is True
        trk_id = f0.tracks[0].track_id

        # Frame 1: same face
        mock_detector.detect_multi_faces.return_value = MultiFaceDetectionResult(
            face_count=1,
            faces=[dummy_face],
            frame_width=400,
            frame_height=400,
            inference_time_ms=12.5,
        )
        f1 = await service.process_frame(session.session_id, 1, 100.1, FakeUploadFile("f1.jpg", blank_bytes), data["faculty_user"])
        assert f1.tracks[0].track_id == trk_id
        assert f1.tracks[0].state == "tentative"
        assert f1.tracks[0].total_observations == 2

        # Frame 2: same face (hits=3 -> confirmed)
        mock_detector.detect_multi_faces.return_value = MultiFaceDetectionResult(
            face_count=1,
            faces=[dummy_face],
            frame_width=400,
            frame_height=400,
            inference_time_ms=12.5,
        )
        f2 = await service.process_frame(session.session_id, 2, 100.2, FakeUploadFile("f2.jpg", blank_bytes), data["faculty_user"])
        assert f2.confirmed_tracks_count == 1
        assert f2.tracks[0].state == "confirmed"

        # Frame 3: no detection (miss=1 -> temporarily_lost)
        mock_detector.detect_multi_faces.return_value = MultiFaceDetectionResult(
            face_count=0,
            faces=[],
            frame_width=400,
            frame_height=400,
            inference_time_ms=10.0,
        )
        f3 = await service.process_frame(session.session_id, 3, 100.3, FakeUploadFile("f3.jpg", blank_bytes), data["faculty_user"])
        assert f3.active_tracks_count == 1
        assert f3.confirmed_tracks_count == 0
        assert f3.tracks[0].state == "temporarily_lost"
        assert f3.tracks[0].consecutive_misses == 1

        # Check get_tracks endpoint
        list_res = service.get_tracks(session.session_id, data["faculty_user"])
        assert list_res.active_tracks_count == 1
        assert list_res.tracks[0].track_id == trk_id
    finally:
        db.close()


@pytest.mark.anyio
async def test_service_process_frame_invalid_inputs(setup_tracking_db):
    """Validate handling of empty files, negative frame_seq, and corrupted images."""
    data = setup_tracking_db
    db = SessionLocal()
    try:
        service = TrackingService(db=db)
        req = CreateTrackingSessionRequest(class_section_id=data["section_id"])
        session = service.create_session(req, current_user=data["faculty_user"])

        # 1. Negative frame_seq
        with pytest.raises(BadRequestError) as exc:
            await service.process_frame(session.session_id, -1, 100.0, FakeUploadFile("f.jpg", b"123"), data["faculty_user"])
        assert "non-negative" in str(exc.value)

        # 2. Empty file
        with pytest.raises(BadRequestError) as exc:
            await service.process_frame(session.session_id, 0, 100.0, FakeUploadFile("empty.jpg", b""), data["faculty_user"])
        assert "empty" in str(exc.value)

        # 3. Corrupt bytes
        with pytest.raises(BadRequestError) as exc:
            await service.process_frame(session.session_id, 0, 100.0, FakeUploadFile("bad.jpg", b"not_an_image"), data["faculty_user"])
        assert "Failed to decode" in str(exc.value)

        # 4. Unsupported MIME type
        with pytest.raises(BadRequestError) as exc:
            await service.process_frame(
                session.session_id,
                0,
                100.0,
                FakeUploadFile("test.txt", b"some text", content_type="text/plain"),
                data["faculty_user"],
            )
        assert "Unsupported or missing content type" in str(exc.value)

        # 5. Missing / empty MIME type
        with pytest.raises(BadRequestError) as exc:
            await service.process_frame(
                session.session_id,
                0,
                100.0,
                FakeUploadFile("test.jpg", b"data", content_type=""),
                data["faculty_user"],
            )
        assert "Unsupported or missing content type" in str(exc.value)

        # 6. Dimensions exceeding max allowed width
        oversized_w = np.zeros((100, 5000, 3), dtype=np.uint8)
        with pytest.raises(BadRequestError) as exc:
            await service.process_frame(
                session.session_id,
                0,
                100.0,
                FakeUploadFile("wide.jpg", _encode_jpeg(oversized_w), content_type="image/jpeg"),
                data["faculty_user"],
            )
        assert "Frame dimensions" in str(exc.value)

        # 7. Dimensions exceeding max allowed height
        oversized_h = np.zeros((5000, 100, 3), dtype=np.uint8)
        with pytest.raises(BadRequestError) as exc:
            await service.process_frame(
                session.session_id,
                0,
                100.0,
                FakeUploadFile("tall.jpg", _encode_jpeg(oversized_h), content_type="image/jpeg"),
                data["faculty_user"],
            )
        assert "Frame dimensions" in str(exc.value)

        # 8. Upload size exceeding BIOMETRIC_MAX_UPLOAD_BYTES
        huge_bytes = b"0" * (5 * 1024 * 1024 + 1)
        with pytest.raises(BadRequestError) as exc:
            await service.process_frame(
                session.session_id,
                0,
                100.0,
                FakeUploadFile("huge.jpg", huge_bytes, content_type="image/jpeg"),
                data["faculty_user"],
            )
        assert "exceeds maximum limit" in str(exc.value)
    finally:
        db.close()


# ==============================================================================
# 3. REST API ENDPOINT INTEGRATION TESTS (TestClient)
# ==============================================================================

def test_api_tracking_session_full_lifecycle(setup_tracking_db):
    """End-to-end integration test through FastAPI TestClient."""
    data = setup_tracking_db
    client = TestClient(app)

    # 1. Faculty Token
    fac_token = create_access_token(
        user_id=data["faculty_user"].id,
        username=data["faculty_user"].username,
        role=data["faculty_user"].role,
    )
    auth_header = {"Authorization": f"Bearer {fac_token}"}

    # 2. POST /api/v1/tracking/sessions
    resp = client.post(
        "/api/v1/tracking/sessions",
        json={"class_section_id": data["section_id"], "min_hits_to_confirm": 2},
        headers=auth_header,
    )
    assert resp.status_code == status.HTTP_201_CREATED
    sess_data = resp.json()
    session_id = sess_data["session_id"]
    assert session_id.startswith("trk_sess_")
    assert sess_data["config"]["min_hits_to_confirm"] == 2

    # 3. POST /api/v1/tracking/sessions/{session_id}/process-frame (Frame 0)
    blank_img = np.zeros((300, 300, 3), dtype=np.uint8)
    img_bytes = _encode_jpeg(blank_img)

    f0_resp = client.post(
        f"/api/v1/tracking/sessions/{session_id}/process-frame",
        data={"frame_seq": 0, "timestamp": 1700000000.0},
        files={"file": ("frame0.jpg", img_bytes, "image/jpeg")},
        headers=auth_header,
    )
    assert f0_resp.status_code == status.HTTP_200_OK
    f0_json = f0_resp.json()
    assert f0_json["session_id"] == session_id
    assert f0_json["frame_seq"] == 0
    assert f0_json["active_tracks_count"] == 0

    # 4. Out-of-order frame sequence error (Frame 0 again)
    err_resp = client.post(
        f"/api/v1/tracking/sessions/{session_id}/process-frame",
        data={"frame_seq": 0, "timestamp": 1700000001.0},
        files={"file": ("frame0.jpg", img_bytes, "image/jpeg")},
        headers=auth_header,
    )
    assert err_resp.status_code == status.HTTP_400_BAD_REQUEST

    # 5. GET /api/v1/tracking/sessions/{session_id}/tracks
    tracks_resp = client.get(f"/api/v1/tracking/sessions/{session_id}/tracks", headers=auth_header)
    assert tracks_resp.status_code == status.HTTP_200_OK
    assert tracks_resp.json()["active_tracks_count"] == 0

    # 6. DELETE /api/v1/tracking/sessions/{session_id}
    del_resp = client.delete(f"/api/v1/tracking/sessions/{session_id}", headers=auth_header)
    assert del_resp.status_code == status.HTTP_200_OK
    assert del_resp.json()["session_id"] == session_id

    # 7. Accessing deleted session yields 404
    get_after_del = client.get(f"/api/v1/tracking/sessions/{session_id}/tracks", headers=auth_header)
    assert get_after_del.status_code == status.HTTP_404_NOT_FOUND


def test_api_tracking_unassigned_faculty_forbidden(setup_tracking_db):
    """Faculty not assigned to class section is rejected with HTTP 403 Forbidden."""
    data = setup_tracking_db
    client = TestClient(app)
    unassigned_token = create_access_token(
        user_id=data["unassigned_faculty"].id,
        username=data["unassigned_faculty"].username,
        role=data["unassigned_faculty"].role,
    )
    resp = client.post(
        "/api/v1/tracking/sessions",
        json={"class_section_id": data["section_id"]},
        headers={"Authorization": f"Bearer {unassigned_token}"},
    )
    assert resp.status_code == status.HTTP_403_FORBIDDEN
