"""
ReliAttend AI — Multi-Face Recognition & 1:N Feature Matching Tests (Phase 8)
Validates 1:N cosine matching, multi-pose max-pooling, ambiguous collision resolution,
duplicate identity conflict handling, class section gallery scoping, revoked consent exclusion,
numerical vector math, error recovery, and zero filesystem footprint.
"""

import io
import os
import uuid
from unittest.mock import MagicMock, patch
import cv2
import numpy as np
import pytest

from app.ai.detector import DetectedFaceBox, QualityMetrics
from app.ai.matcher import CandidateMatch, GalleryItem, VectorMatcher
from app.ai.recognition import FaceRecognizer
from app.config import get_settings
from app.core.exceptions import AuthorizationForbiddenError, BadRequestError, ResourceNotFoundError
from app.database import Base, SessionLocal, engine
from app.models.academic import AcademicYear, ClassSection, Department, FacultyAssignment, Programme, Semester
from app.models.student import ConsentRecord, FaceReference, Student
from app.models.user import User
from app.services.recognition_service import RecognitionService

settings = get_settings()


def _random_normalized_embedding(seed: int = 42) -> np.ndarray:
    """Generate deterministic L2-normalized 128-d float32 vector."""
    rng = np.random.RandomState(seed)
    v = rng.randn(128).astype(np.float32)
    return v / np.linalg.norm(v)


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


# ==============================================================================
# 1. VECTOR MATCHER NUMERICAL INTEGRITY & MATHEMATICAL PROPERTIES
# ==============================================================================

def test_vector_matcher_orthogonal_and_identical_embeddings():
    """Verify mathematical cosine similarity properties: identical = 1.0, orthogonal = 0.0."""
    v1 = _random_normalized_embedding(10)
    # Identical vector
    gallery = [
        GalleryItem(
            reference_id=1,
            student_id=101,
            student_str_id="STU_101",
            student_name="Student 101",
            pose_label="frontal",
            is_primary=True,
            embedding=v1,
        )
    ]
    matcher = VectorMatcher(gallery, similarity_threshold=0.65)
    res = matcher.match_embeddings([(1, v1)])

    assert len(res) == 1
    assert res[0].match_status == "matched"
    assert res[0].top_match.student_id == 101
    assert pytest.approx(res[0].top_match.similarity_score, 0.0001) == 1.0
    # Single candidate in gallery: margin_to_second must be None
    assert res[0].margin_to_second is None

    # Orthogonal vector (Gram-Schmidt projection)
    rng = np.random.RandomState(99)
    rand_v = rng.randn(128).astype(np.float32)
    ortho_v = rand_v - np.dot(rand_v, v1) * v1
    ortho_v = ortho_v / np.linalg.norm(ortho_v)
    assert abs(float(np.dot(v1, ortho_v))) < 1e-5

    res_ortho = matcher.match_embeddings([(2, ortho_v)])
    assert res_ortho[0].match_status == "unrecognized"
    assert abs(res_ortho[0].top_candidates[0].similarity_score) < 1e-4
    # Single candidate in gallery: margin_to_second must be None even when unrecognized
    assert res_ortho[0].margin_to_second is None


def test_vector_matcher_multi_pose_max_pooling():
    """Student with 2 poses (frontal, slight_left); query matches slight_left best."""
    v_frontal = _random_normalized_embedding(1)
    v_left = _random_normalized_embedding(2)

    gallery = [
        GalleryItem(
            reference_id=1,
            student_id=201,
            student_str_id="STU_201",
            student_name="Student 201",
            pose_label="frontal",
            is_primary=True,
            embedding=v_frontal,
        ),
        GalleryItem(
            reference_id=2,
            student_id=201,
            student_str_id="STU_201",
            student_name="Student 201",
            pose_label="slight_left",
            is_primary=False,
            embedding=v_left,
        ),
    ]

    matcher = VectorMatcher(gallery, similarity_threshold=0.65)
    # Query is close to slight_left
    query_v = 0.95 * v_left + 0.05 * v_frontal
    query_v = query_v / np.linalg.norm(query_v)

    res = matcher.match_embeddings([(1, query_v)])
    assert res[0].match_status == "matched"
    assert res[0].top_match.matched_pose == "slight_left"
    assert res[0].top_match.similarity_score > 0.90


def test_vector_matcher_ambiguity_margin_resolution():
    """Top-1 and Top-2 score difference < 0.05 must result in status 'uncertain'."""
    v1 = _random_normalized_embedding(10)
    # v2 is very close to v1 (small perturbation)
    v2 = v1 + 0.03 * np.random.RandomState(42).randn(128).astype(np.float32)
    v2 = v2 / np.linalg.norm(v2)

    gallery = [
        GalleryItem(
            reference_id=1,
            student_id=1,
            student_str_id="STU_001",
            student_name="Student 1",
            pose_label="frontal",
            is_primary=True,
            embedding=v1,
        ),
        GalleryItem(
            reference_id=2,
            student_id=2,
            student_str_id="STU_002",
            student_name="Student 2",
            pose_label="frontal",
            is_primary=True,
            embedding=v2,
        ),
    ]

    matcher = VectorMatcher(gallery, similarity_threshold=0.65, ambiguity_margin=0.05)
    query_v = (v1 + v2) / np.linalg.norm(v1 + v2)

    res = matcher.match_embeddings([(1, query_v)])
    assert res[0].match_status == "uncertain"
    assert "Ambiguous match" in res[0].reason
    assert res[0].margin_to_second < 0.05


def test_vector_matcher_duplicate_identity_conflict_flagging():
    """If 2 detected faces in the same frame match the same student, both are flagged as uncertain."""
    v_target = _random_normalized_embedding(50)

    gallery = [
        GalleryItem(
            reference_id=1,
            student_id=505,
            student_str_id="STU_505",
            student_name="Student 505",
            pose_label="frontal",
            is_primary=True,
            embedding=v_target,
        ),
    ]

    matcher = VectorMatcher(gallery, similarity_threshold=0.65)
    # Two distinct faces in the same frame both matching STU_505 strongly
    res = matcher.match_embeddings([(1, v_target), (2, v_target)])

    assert len(res) == 2
    assert res[0].match_status == "uncertain"
    assert res[1].match_status == "uncertain"
    assert "Duplicate identity conflict" in res[0].reason
    assert "Duplicate identity conflict" in res[1].reason


# ==============================================================================
# 2. END-TO-END SERVICE & DATABASE SCOPING TESTS
# ==============================================================================

@pytest.fixture
def setup_recognition_db():
    """Sets up an isolated class section with enrolled students, consent, and embeddings."""
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    run_id = uuid.uuid4().hex[:6]

    # Faculty & Section
    faculty = User(
        email=f"fac_p8_{run_id}@reliattend.com",
        username=f"fac_p8_{run_id}",
        password_hash="hashed",
        full_name="Phase 8 Faculty",
        role="faculty",
        is_active=True,
    )
    db.add(faculty)
    db.commit()

    sec = ClassSection(name=f"Sec P8 {run_id}", semester_id=1, max_students=60, is_active=True)
    db.add(sec)
    db.commit()

    # Assignment
    assignment = FacultyAssignment(
        user_id=faculty.id,
        subject_id=1,
        class_section_id=sec.id,
        assigned_by=faculty.id,
        is_active=True,
    )
    db.add(assignment)

    # Student 1: Enrolled, Active Consent, Reference Embedding
    stu1 = Student(
        student_id=f"STU_P8_1_{run_id}",
        name=f"Alice P8 {run_id}",
        roll_number=f"ROLL_P8_1_{run_id}",
        class_section_id=sec.id,
        registration_status="enrolled",
        is_active=True,
    )
    db.add(stu1)
    db.commit()

    consent1 = ConsentRecord(
        student_id=stu1.id,
        consent_type="institutional_enrollment",
        is_active=True,
    )
    db.add(consent1)

    v1 = _random_normalized_embedding(101)
    ref1 = FaceReference(
        student_id=stu1.id,
        image_path=f"storage/ref_{run_id}_1.jpg",
        embedding=v1.tobytes(),
        embedding_size=128,
        model_name="opencv_sface",
        model_version="2021dec",
        pose_label="frontal",
        is_primary=True,
        status="active",
    )
    db.add(ref1)

    # Student 2: Consent Revoked/Inactive (Should be excluded from gallery)
    stu2 = Student(
        student_id=f"STU_P8_2_{run_id}",
        name=f"Bob P8 {run_id}",
        roll_number=f"ROLL_P8_2_{run_id}",
        class_section_id=sec.id,
        registration_status="enrolled",
        is_active=True,
    )
    db.add(stu2)
    db.commit()

    consent2 = ConsentRecord(
        student_id=stu2.id,
        consent_type="institutional_enrollment",
        is_active=False,  # REVOKED
    )
    db.add(consent2)

    v2 = _random_normalized_embedding(102)
    ref2 = FaceReference(
        student_id=stu2.id,
        image_path=f"storage/ref_{run_id}_2.jpg",
        embedding=v2.tobytes(),
        embedding_size=128,
        model_name="opencv_sface",
        model_version="2021dec",
        pose_label="frontal",
        is_primary=True,
        status="active",
    )
    db.add(ref2)

    # Other Section Student (Should be excluded by section boundary)
    other_sec = ClassSection(name=f"Other Sec {run_id}", semester_id=1, max_students=60, is_active=True)
    db.add(other_sec)
    db.commit()

    stu_other = Student(
        student_id=f"STU_OTHER_{run_id}",
        name=f"Charlie P8 {run_id}",
        roll_number=f"ROLL_OTHER_{run_id}",
        class_section_id=other_sec.id,
        registration_status="enrolled",
        is_active=True,
    )
    db.add(stu_other)
    db.commit()

    consent_other = ConsentRecord(student_id=stu_other.id, consent_type="institutional_enrollment", is_active=True)
    db.add(consent_other)

    v_other = _random_normalized_embedding(103)
    ref_other = FaceReference(
        student_id=stu_other.id,
        image_path=f"storage/ref_other_{run_id}.jpg",
        embedding=v_other.tobytes(),
        embedding_size=128,
        model_name="opencv_sface",
        model_version="2021dec",
        pose_label="frontal",
        is_primary=True,
        status="active",
    )
    db.add(ref_other)

    db.commit()

    yield {
        "db": db,
        "faculty": faculty,
        "section_id": sec.id,
        "stu1_id": stu1.student_id,
        "v1": v1,
        "v2": v2,
        "v_other": v_other,
        "other_section_id": other_sec.id,
    }

    db.close()


@pytest.mark.anyio
async def test_gallery_scoping_and_revoked_consent_exclusion(setup_recognition_db):
    """Verify that only enrolled students with active consent in the targeted section are in the gallery."""
    ctx = setup_recognition_db
    service = RecognitionService(db=ctx["db"])

    gallery, eligible_count = service._load_section_gallery(ctx["section_id"])

    # Only stu1 should be present:
    # - stu2 excluded due to inactive consent
    # - stu_other excluded due to different section
    assert eligible_count == 1
    assert len(gallery) == 1
    assert gallery[0].student_str_id == ctx["stu1_id"]


@pytest.mark.anyio
async def test_recognize_frame_strong_match(setup_recognition_db):
    """End-to-end test recognizing a detected face in a frame against section gallery."""
    ctx = setup_recognition_db
    service = RecognitionService(db=ctx["db"])

    img = np.full((300, 300, 3), 120, dtype=np.uint8)
    upload = FakeUploadFile("test_frame.jpg", _encode_jpeg(img))

    # Mock detection with 1 face box
    fake_face = DetectedFaceBox(
        face_id=1,
        bounding_box=(50, 50, 100, 100),
        confidence=0.95,
        landmarks={"right_eye": (70, 70), "left_eye": (110, 70), "nose_tip": (90, 90), "mouth_right": (75, 120), "mouth_left": (105, 120)},
        quality=QualityMetrics(100.0, 100.0, 100, 100, 0.95, True, True, True),
        is_quality_acceptable=True,
        face_data=np.array([50, 50, 100, 100, 70, 70, 110, 70, 90, 90, 75, 120, 105, 120, 0.95], dtype=np.float32),
    )

    with patch.object(service.detector, "detect_multi_faces") as mock_det:
        mock_det.return_value = MagicMock(faces=[fake_face], is_scaled=False, scale_factor=1.0)
        with patch.object(service.recognizer, "extract_embedding") as mock_emb:
            mock_emb.return_value = MagicMock(is_valid=True, embedding_bytes=ctx["v1"].tobytes())

            resp = await service.recognize_frame_for_section(
                class_section_id=ctx["section_id"],
                file=upload,
                current_user=ctx["faculty"],
            )

            assert resp.success is True
            assert resp.data.summary.faces_detected == 1
            assert resp.data.summary.faces_recognized == 1
            assert resp.data.summary.faces_unrecognized == 0
            assert resp.data.summary.faces_uncertain == 0
            f = resp.data.faces[0]
            assert f.match_status == "matched"
            assert f.top_match.student_id == ctx["stu1_id"]
            assert pytest.approx(f.top_match.similarity_score, 0.001) == 1.0


@pytest.mark.anyio
async def test_unauthorized_faculty_rejected(setup_recognition_db):
    """Faculty trying to access an unassigned section must be rejected with 403 Forbidden."""
    ctx = setup_recognition_db
    service = RecognitionService(db=ctx["db"])
    upload = FakeUploadFile("test.jpg", _encode_jpeg(np.full((100, 100, 3), 120, dtype=np.uint8)))

    # Faculty is NOT assigned to other_section_id
    with pytest.raises(AuthorizationForbiddenError) as exc_info:
        await service.recognize_frame_for_section(
            class_section_id=ctx["other_section_id"],
            file=upload,
            current_user=ctx["faculty"],
        )
    assert "not assigned" in str(exc_info.value)


@pytest.mark.anyio
async def test_zero_faces_frame_returns_200_empty(setup_recognition_db):
    """Frame with zero detected faces returns HTTP 200 with 0 recognized and empty faces array."""
    ctx = setup_recognition_db
    service = RecognitionService(db=ctx["db"])
    upload = FakeUploadFile("empty.jpg", _encode_jpeg(np.full((100, 100, 3), 120, dtype=np.uint8)))

    with patch.object(service.detector, "detect_multi_faces") as mock_det:
        mock_det.return_value = MagicMock(faces=[], is_scaled=False, scale_factor=1.0)

        resp = await service.recognize_frame_for_section(
            class_section_id=ctx["section_id"],
            file=upload,
            current_user=ctx["faculty"],
        )

        assert resp.success is True
        assert resp.data.summary.faces_detected == 0
        assert resp.data.summary.faces_recognized == 0
        assert resp.data.faces == []


@pytest.mark.anyio
async def test_zero_filesystem_persistence(setup_recognition_db):
    """Recognition must not leave any files or crops in local storage."""
    ctx = setup_recognition_db
    service = RecognitionService(db=ctx["db"])
    storage_dir = settings.BIOMETRIC_STORAGE_DIR
    files_before = set(os.listdir(storage_dir)) if os.path.exists(storage_dir) else set()

    upload = FakeUploadFile("frame.jpg", _encode_jpeg(np.full((200, 200, 3), 120, dtype=np.uint8)))

    with patch.object(service.detector, "detect_multi_faces") as mock_det:
        mock_det.return_value = MagicMock(faces=[], is_scaled=False, scale_factor=1.0)
        await service.recognize_frame_for_section(
            class_section_id=ctx["section_id"],
            file=upload,
            current_user=ctx["faculty"],
        )

    files_after = set(os.listdir(storage_dir)) if os.path.exists(storage_dir) else set()
    assert files_before == files_after


# ==============================================================================
# 3. ADVANCED RECOGNITION COVERAGE & MARGIN REFINEMENT TESTS
# Note: All vectors and scores used below are synthetic test data for boundary
# validation, not claims of real-world operational recognition accuracy.
# ==============================================================================

def test_margin_to_second_null_when_single_student_and_json_serializes_to_null():
    """Requirement 1, 4, 5: Return margin_to_second: null when fewer than 2 candidates exist."""
    v1 = _random_normalized_embedding(11)
    gallery = [
        GalleryItem(
            reference_id=1,
            student_id=101,
            student_str_id="STU_101",
            student_name="Student 101",
            pose_label="frontal",
            is_primary=True,
            embedding=v1,
        )
    ]
    matcher = VectorMatcher(gallery, similarity_threshold=0.65)
    res = matcher.match_embeddings([(1, v1)])

    assert res[0].margin_to_second is None

    # Check Pydantic JSON serialization turns None into JSON null
    from app.schemas.recognition import RecognizedFaceItemSchema
    from app.schemas.detection import BoundingBoxSchema, FaceQualitySchema
    item = RecognizedFaceItemSchema(
        face_id=1,
        bounding_box=BoundingBoxSchema(x=10, y=10, width=50, height=50),
        detection_confidence=0.95,
        quality=FaceQualitySchema(blur_score=100, brightness_score=100, face_width=50, face_height=50, is_blur_acceptable=True, is_brightness_acceptable=True, is_resolution_acceptable=True, is_acceptable=True),
        match_status=res[0].match_status,
        top_match=None,
        top_candidates=[],
        margin_to_second=res[0].margin_to_second,
    )
    dumped = item.model_dump()
    assert dumped["margin_to_second"] is None
    import json
    json_str = item.model_dump_json()
    assert '"margin_to_second":null' in json_str or '"margin_to_second": null' in json_str


def test_margin_computed_using_highest_scoring_reference_per_student():
    """Requirement 2: When multiple references exist per student, margin is computed across distinct students."""
    # Student 1 has 2 references: v1_a and v1_b
    v1_a = _random_normalized_embedding(21)
    v1_b = _random_normalized_embedding(22)
    # Student 2 has 1 reference: v2
    v2 = _random_normalized_embedding(23)

    gallery = [
        GalleryItem(reference_id=1, student_id=1, student_str_id="STU_1", student_name="S1", pose_label="frontal", is_primary=True, embedding=v1_a),
        GalleryItem(reference_id=2, student_id=1, student_str_id="STU_1", student_name="S1", pose_label="slight_left", is_primary=False, embedding=v1_b),
        GalleryItem(reference_id=3, student_id=2, student_str_id="STU_2", student_name="S2", pose_label="frontal", is_primary=True, embedding=v2),
    ]

    matcher = VectorMatcher(gallery, similarity_threshold=0.65, ambiguity_margin=0.05)
    # Query is identical to v1_b
    res = matcher.match_embeddings([(1, v1_b)])

    assert len(res) == 1
    assert res[0].match_status == "matched"
    assert res[0].top_match.student_id == 1
    # Only 2 distinct students (ID 1 and ID 2); top_candidates length must be 2, not 3
    assert len(res[0].top_candidates) == 2
    assert res[0].top_candidates[0].student_id == 1
    assert res[0].top_candidates[1].student_id == 2
    # Margin must be difference between Student 1's best score and Student 2's best score
    expected_margin = round(res[0].top_candidates[0].similarity_score - res[0].top_candidates[1].similarity_score, 4)
    assert res[0].margin_to_second == expected_margin
    assert res[0].margin_to_second is not None


def test_enrolled_student_alongside_unknown_persons():
    """Mixed frame: 1 enrolled student (S >= 0.65) and 2 unknown non-enrolled persons (S < 0.65)."""
    v_enrolled = _random_normalized_embedding(31)
    # Uncorrelated random vectors representing unknown persons
    v_unknown1 = _random_normalized_embedding(88)
    v_unknown2 = _random_normalized_embedding(99)

    gallery = [
        GalleryItem(reference_id=1, student_id=301, student_str_id="STU_301", student_name="Enrolled Alice", pose_label="frontal", is_primary=True, embedding=v_enrolled),
    ]

    matcher = VectorMatcher(gallery, similarity_threshold=0.65)
    res = matcher.match_embeddings([(1, v_enrolled), (2, v_unknown1), (3, v_unknown2)])

    assert len(res) == 3
    # Face 1 is enrolled
    assert res[0].match_status == "matched"
    assert res[0].top_match.student_str_id == "STU_301"

    # Face 2 is unknown person
    assert res[1].match_status == "unrecognized"
    assert res[1].top_match is None
    assert res[1].top_candidates[0].similarity_score < 0.65

    # Face 3 is unknown person
    assert res[2].match_status == "unrecognized"
    assert res[2].top_match is None
    assert res[2].top_candidates[0].similarity_score < 0.65


def test_multiple_enrolled_students_simultaneously_present():
    """Frame with 2 enrolled students simultaneously present: both must be recognized distinctly."""
    v_alice = _random_normalized_embedding(41)
    v_bob = _random_normalized_embedding(42)

    gallery = [
        GalleryItem(reference_id=1, student_id=401, student_str_id="STU_401", student_name="Alice", pose_label="frontal", is_primary=True, embedding=v_alice),
        GalleryItem(reference_id=2, student_id=402, student_str_id="STU_402", student_name="Bob", pose_label="frontal", is_primary=True, embedding=v_bob),
    ]

    matcher = VectorMatcher(gallery, similarity_threshold=0.65, ambiguity_margin=0.05)
    res = matcher.match_embeddings([(1, v_alice), (2, v_bob)])

    assert len(res) == 2
    # Face 1 recognized as Alice
    assert res[0].match_status == "matched"
    assert res[0].top_match.student_str_id == "STU_401"
    assert res[0].margin_to_second is not None
    assert res[0].margin_to_second > 0.05

    # Face 2 recognized as Bob
    assert res[1].match_status == "matched"
    assert res[1].top_match.student_str_id == "STU_402"
    assert res[1].margin_to_second is not None
    assert res[1].margin_to_second > 0.05

