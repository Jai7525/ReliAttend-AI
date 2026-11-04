"""
ReliAttend AI — Multi-Face Detection Automated Tests (Phase 7)
Unit & Service level tests validating zero, one, two, and dense faces detection,
quality checks, coordinate scaling accuracy, security validations, and zero filesystem footprint.
"""

import io
import os
import threading
from unittest.mock import MagicMock, patch
import cv2
import numpy as np
import pytest

from app.ai.detector import (
    DetectedFaceBox,
    FaceDetector,
    MultiFaceDetectionResult,
    QualityMetrics,
)
from app.config import get_settings
from app.core.exceptions import BadRequestError
from app.models.user import User
from app.services.detection_service import DetectionService

settings = get_settings()


def _encode_jpeg(img_bgr: np.ndarray) -> bytes:
    """Encode OpenCV BGR image into JPEG bytes."""
    _, buf = cv2.imencode(".jpg", img_bgr)
    return buf.tobytes()


def _encode_png(img_bgr: np.ndarray) -> bytes:
    """Encode OpenCV BGR image into PNG bytes."""
    _, buf = cv2.imencode(".png", img_bgr)
    return buf.tobytes()


class FakeUploadFile:
    """Synchronous mock of FastAPI UploadFile for fast unit testing."""

    def __init__(self, filename: str, content: bytes, content_type: str):
        self.filename = filename
        self.content = content
        self.content_type = content_type

    async def read(self) -> bytes:
        return self.content


# ==============================================================================
# 1. ZERO FACES TESTS (HTTP 200 with empty list)
# ==============================================================================

@pytest.mark.anyio
async def test_detect_zero_faces_returns_empty_list():
    """A frame containing zero faces must return 0 faces and empty list, not an error."""
    service = DetectionService()
    img = np.full((400, 600, 3), 128, dtype=np.uint8)
    upload = FakeUploadFile(filename="empty_room.jpg", content=_encode_jpeg(img), content_type="image/jpeg")
    user = User(id=1, username="faculty_user", role="faculty")

    res = await service.detect_faces_in_frame(upload, user)

    assert res.success is True
    assert res.data.face_count == 0
    assert len(res.data.faces) == 0
    assert res.data.frame_metadata.width == 600
    assert res.data.frame_metadata.height == 400


# ==============================================================================
# 2. MULTI-FACE DETECTION TESTS (1, 2, 4+ Faces)
# ==============================================================================

def test_detect_single_face_via_service():
    """Unit test single face detection on detector engine."""
    yunet_path = os.path.join(settings.BIOMETRIC_MODELS_DIR, "face_detection_yunet_2023mar.onnx")
    detector = FaceDetector(model_path=yunet_path)

    fake_face = np.array([50, 60, 100, 120, 70, 80, 110, 80, 90, 110, 75, 140, 105, 140, 0.95], dtype=np.float32)
    fake_img = np.full((300, 300, 3), 150, dtype=np.uint8)

    with patch("cv2.FaceDetectorYN.detect", return_value=(None, np.array([fake_face]))):
        res = detector.detect_multi_faces(fake_img)
        assert res.face_count == 1
        assert len(res.faces) == 1
        f = res.faces[0]
        assert f.bounding_box == (50, 60, 100, 120)
        assert f.confidence == 0.95
        assert f.landmarks["right_eye"] == (70, 80)
        assert f.landmarks["left_eye"] == (110, 80)
        assert f.landmarks["nose_tip"] == (90, 110)
        assert f.landmarks["mouth_right"] == (75, 140)
        assert f.landmarks["mouth_left"] == (105, 140)


@pytest.mark.anyio
async def test_detect_multiple_faces_via_service():
    """Test detecting 2 faces with distinct bounding boxes and landmarks."""
    fake_face1 = np.array([30, 40, 80, 90, 45, 60, 75, 60, 60, 80, 50, 100, 70, 100, 0.92], dtype=np.float32)
    fake_face2 = np.array([160, 50, 85, 95, 175, 70, 205, 70, 190, 90, 180, 110, 200, 110, 0.88], dtype=np.float32)

    fake_img = np.full((400, 400, 3), 120, dtype=np.uint8)
    upload = FakeUploadFile(filename="group_shot.jpg", content=_encode_jpeg(fake_img), content_type="image/jpeg")
    user = User(id=1, username="faculty_user", role="faculty")
    service = DetectionService()

    with patch("cv2.FaceDetectorYN.detect", return_value=(None, np.array([fake_face1, fake_face2]))):
        res = await service.detect_faces_in_frame(upload, user)
        assert res.data.face_count == 2
        faces = res.data.faces
        assert len(faces) == 2
        assert faces[0].face_id == 1
        assert faces[0].confidence == 0.92
        assert faces[1].face_id == 2
        assert faces[1].confidence == 0.88


@pytest.mark.anyio
async def test_detect_dense_classroom_four_faces():
    """Verify handling of 4 or more faces in a single frame."""
    fake_faces = [
        np.array([20 * i, 30, 70, 80, 20 * i + 15, 50, 20 * i + 45, 50, 20 * i + 30, 65, 20 * i + 20, 80, 20 * i + 40, 80, 0.90 - i * 0.05], dtype=np.float32)
        for i in range(1, 5)
    ]
    fake_img = np.full((400, 600, 3), 140, dtype=np.uint8)
    upload = FakeUploadFile(filename="dense_classroom.png", content=_encode_png(fake_img), content_type="image/png")
    user = User(id=2, username="admin_user", role="admin")
    service = DetectionService()

    with patch("cv2.FaceDetectorYN.detect", return_value=(None, np.array(fake_faces))):
        res = await service.detect_faces_in_frame(upload, user)
        assert res.data.face_count == 4
        assert len(res.data.faces) == 4


# ==============================================================================
# 3. COORDINATE TRANSFORMATION & SCALING TESTS
# ==============================================================================

def test_inference_rescaling_transforms_coordinates_to_original():
    """Verify that if an image exceeds max inference dimension, bboxes are scaled back to original."""
    yunet_path = os.path.join(settings.BIOMETRIC_MODELS_DIR, "face_detection_yunet_2023mar.onnx")
    detector = FaceDetector(model_path=yunet_path)

    # 3840x2160 image with max inference dimension 1920 (scale factor = 0.5)
    large_img = np.full((2160, 3840, 3), 120, dtype=np.uint8)

    # Detection in scaled space (1920x1080)
    scaled_face = np.array([100, 100, 50, 60, 110, 120, 130, 120, 120, 135, 115, 150, 135, 150, 0.90], dtype=np.float32)

    with patch("cv2.FaceDetectorYN.detect", return_value=(None, np.array([scaled_face]))):
        res = detector.detect_multi_faces(large_img, max_inference_dimension=1920)
        assert res.is_scaled is True
        assert res.scale_factor == 0.5
        assert res.frame_width == 3840
        assert res.frame_height == 2160
        f = res.faces[0]
        # In original 3840x2160 space, coordinates should be doubled (inv_scale = 2.0)
        assert f.bounding_box[0] == 200
        assert f.bounding_box[1] == 200
        assert f.bounding_box[2] == 100
        assert f.bounding_box[3] == 120
        assert f.landmarks["right_eye"] == (220, 240)


# ==============================================================================
# 4. QUALITY VS CONFIDENCE INDEPENDENCE
# ==============================================================================

@pytest.mark.anyio
async def test_poor_quality_faces_still_reported_with_quality_flags():
    """Verify that poorly lit or blurry faces are still detected, with quality.is_acceptable=False."""
    fake_face = np.array([10, 10, 60, 60, 25, 25, 45, 25, 35, 40, 25, 50, 45, 50, 0.95], dtype=np.float32)
    # Uniform pitch black image (brightness = 0.0 < 40.0 min brightness)
    dark_img = np.zeros((200, 200, 3), dtype=np.uint8)
    upload = FakeUploadFile(filename="dark_frame.jpg", content=_encode_jpeg(dark_img), content_type="image/jpeg")
    user = User(id=1, username="faculty_user", role="faculty")
    service = DetectionService()

    with patch("cv2.FaceDetectorYN.detect", return_value=(None, np.array([fake_face]))):
        res = await service.detect_faces_in_frame(upload, user)
        assert res.data.face_count == 1
        f = res.data.faces[0]
        assert f.confidence == 0.95
        assert f.quality.is_brightness_acceptable is False
        assert f.quality.is_acceptable is False


# ==============================================================================
# 5. SECURITY, VALIDATION & REJECTIONS
# ==============================================================================

@pytest.mark.anyio
async def test_reject_corrupted_image():
    """Uploading corrupt or truncated byte stream raises BadRequestError."""
    corrupted_bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF" + b"\x00" * 30  # Incomplete JPEG
    upload = FakeUploadFile(filename="corrupt.jpg", content=corrupted_bytes, content_type="image/jpeg")
    user = User(id=1, username="faculty_user", role="faculty")
    service = DetectionService()

    with pytest.raises(BadRequestError) as exc_info:
        await service.detect_faces_in_frame(upload, user)
    assert "Failed to decode image" in str(exc_info.value)


@pytest.mark.anyio
async def test_reject_unsupported_file_type():
    """Uploading a text or non-image file raises BadRequestError."""
    upload = FakeUploadFile(filename="notes.txt", content=b"plain text data", content_type="text/plain")
    user = User(id=1, username="faculty_user", role="faculty")
    service = DetectionService()

    with pytest.raises(BadRequestError) as exc_info:
        await service.detect_faces_in_frame(upload, user)
    assert "Unsupported file type" in str(exc_info.value)


@pytest.mark.anyio
async def test_reject_oversized_file():
    """Files exceeding 5MB are rejected with BadRequestError."""
    oversized_bytes = b"\xff\xd8\xff" + b"\x00" * (5 * 1024 * 1024 + 100)
    upload = FakeUploadFile(filename="huge.jpg", content=oversized_bytes, content_type="image/jpeg")
    user = User(id=1, username="faculty_user", role="faculty")
    service = DetectionService()

    with pytest.raises(BadRequestError) as exc_info:
        await service.detect_faces_in_frame(upload, user)
    assert "exceeds maximum allowed limit" in str(exc_info.value)


# ==============================================================================
# 6. ZERO FILESYSTEM PERSISTENCE CHECK
# ==============================================================================

@pytest.mark.anyio
async def test_zero_disk_persistence():
    """Ensure multi-face detection leaves NO files in the filesystem."""
    storage_dir = settings.BIOMETRIC_STORAGE_DIR
    files_before = set(os.listdir(storage_dir)) if os.path.exists(storage_dir) else set()

    img = np.full((200, 200, 3), 128, dtype=np.uint8)
    upload = FakeUploadFile(filename="temp_frame.jpg", content=_encode_jpeg(img), content_type="image/jpeg")
    user = User(id=1, username="faculty_user", role="faculty")
    service = DetectionService()

    res = await service.detect_faces_in_frame(upload, user)
    assert res.success is True

    files_after = set(os.listdir(storage_dir)) if os.path.exists(storage_dir) else set()
    assert files_before == files_after, "Detection must not persist files to storage!"


# ==============================================================================
# 7. THREAD-SAFETY & CONCURRENT DETECTOR ACCESS
# ==============================================================================

def test_concurrent_detector_calls_thread_safety():
    """Verify that multiple concurrent threads calling detect_multi_faces do not corrupt state."""
    yunet_path = os.path.join(settings.BIOMETRIC_MODELS_DIR, "face_detection_yunet_2023mar.onnx")
    detector = FaceDetector(model_path=yunet_path)

    img1 = np.full((300, 400, 3), 128, dtype=np.uint8)
    img2 = np.full((500, 350, 3), 128, dtype=np.uint8)

    errors = []

    def run_worker(img, expected_w, expected_h):
        try:
            for _ in range(5):
                res = detector.detect_multi_faces(img)
                assert res.frame_width == expected_w
                assert res.frame_height == expected_h
        except Exception as e:
            errors.append(e)

    t1 = threading.Thread(target=run_worker, args=(img1, 400, 300))
    t2 = threading.Thread(target=run_worker, args=(img2, 350, 500))

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert len(errors) == 0, f"Thread safety violated: {errors}"
