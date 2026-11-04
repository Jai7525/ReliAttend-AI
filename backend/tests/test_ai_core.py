"""
ReliAttend AI — Unit Tests for AI Core (Face Detection, Quality Checks & Feature Extraction)
Tests detector.py and recognition.py in total isolation without hitting database or network.
Uses synthetic geometric test patterns and real YuNet/SFace ONNX weights.
All test image generation stays in-memory (no biometric images saved to git).
"""

import cv2
import numpy as np
import pytest
from app.ai.detector import FaceDetector, QualityMetrics
from app.ai.recognition import FaceRecognizer


YUNET_MODEL = "backend/models/face_detection_yunet_2023mar.onnx"
SFACE_MODEL = "backend/models/face_recognition_sface_2021dec.onnx"


@pytest.fixture(scope="module")
def detector():
    return FaceDetector(
        model_path=YUNET_MODEL,
        confidence_threshold=0.85,
        min_face_width=80,
        min_face_height=80,
        blur_threshold=50.0,
        min_brightness=40.0,
        max_brightness=220.0,
    )


@pytest.fixture(scope="module")
def recognizer():
    return FaceRecognizer(model_path=SFACE_MODEL)


# --- 1. Image Decoding Safety Tests ---

def test_decode_invalid_corrupt_bytes_returns_none(detector):
    assert detector.decode_image_bytes(b"") is None
    assert detector.decode_image_bytes(b"not an image at all") is None
    assert detector.decode_image_bytes(b"\x00\x00\x00\x00") is None


def test_decode_valid_synthetic_png_succeeds(detector):
    # Create 200x200 blank synthetic image and encode to PNG
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    img[:] = (120, 120, 120)
    success, encoded = cv2.imencode(".png", img)
    assert success is True

    decoded = detector.decode_image_bytes(encoded.tobytes())
    assert decoded is not None
    assert decoded.shape == (200, 200, 3)


# --- 2. Quality Metrics Unit Tests ---

def test_quality_metrics_dark_image_rejected(detector):
    # Very dark image (mean brightness < 40)
    dark_img = np.ones((100, 100, 3), dtype=np.uint8) * 10
    metrics = detector.compute_quality_metrics(dark_img, 0, 0, 100, 100, 0.95)
    assert metrics.is_brightness_acceptable is False
    assert metrics.brightness_score < 40.0


def test_quality_metrics_bright_image_rejected(detector):
    # Washed out image (mean brightness > 220)
    bright_img = np.ones((100, 100, 3), dtype=np.uint8) * 245
    metrics = detector.compute_quality_metrics(bright_img, 0, 0, 100, 100, 0.95)
    assert metrics.is_brightness_acceptable is False
    assert metrics.brightness_score > 220.0


def test_quality_metrics_blurry_image_rejected(detector):
    # Completely uniform gray square (no edge variation -> Laplacian variance ~ 0.0)
    flat_img = np.ones((100, 100, 3), dtype=np.uint8) * 128
    metrics = detector.compute_quality_metrics(flat_img, 0, 0, 100, 100, 0.95)
    assert metrics.is_blur_acceptable is False
    assert metrics.blur_score < detector.blur_threshold


def test_quality_metrics_low_resolution_rejected(detector):
    # Face crop width < 80 px
    img = np.ones((100, 100, 3), dtype=np.uint8) * 128
    metrics = detector.compute_quality_metrics(img, 10, 10, 50, 50, 0.95)
    assert metrics.is_resolution_acceptable is False
    assert metrics.face_width == 50


# --- 3. Face Count Constraint Tests ---

def test_detect_zero_faces_on_blank_canvas(detector):
    blank = np.zeros((300, 300, 3), dtype=np.uint8)
    result = detector.detect_and_validate(blank)
    assert result.is_valid is False
    assert result.face_count == 0
    assert "No face detected" in result.error_message


# --- 4. SFace Feature Extraction & Embedding Validation Tests ---

def test_extract_embedding_dimension_and_serialization(recognizer):
    # Synthetic realistic aligned face chip (112x112)
    synthetic_face = np.random.randint(60, 200, (112, 112, 3), dtype=np.uint8)
    # Synthetic mock 15-element YuNet detection row for landmarks
    mock_face_data = np.array([
        10.0, 10.0, 90.0, 90.0,  # bbox x, y, w, h
        30.0, 40.0, 70.0, 40.0,  # right eye, left eye
        50.0, 60.0,              # nose tip
        35.0, 80.0, 65.0, 80.0,  # mouth corners
        0.98                      # confidence
    ], dtype=np.float32)

    result = recognizer.extract_embedding(synthetic_face, mock_face_data)
    assert result.is_valid is True
    assert result.embedding_dimension == 128
    assert result.model_name == "opencv_sface"
    assert result.model_version == "2021dec"
    assert result.preprocessing_version == "sface_aligned_112x112_v1"
    # Serialized bytes length: 128 float32 values = 512 bytes
    assert len(result.embedding_bytes) == 512

    # Deserialization test
    unpacked_vector = recognizer.deserialize_embedding(result.embedding_bytes)
    assert isinstance(unpacked_vector, np.ndarray)
    assert unpacked_vector.shape == (128,)
    assert unpacked_vector.dtype == np.float32
    assert np.all(np.isfinite(unpacked_vector))

    # Verify L2 normalization (unit length norm ~= 1.0)
    norm = np.linalg.norm(unpacked_vector)
    assert pytest.approx(float(norm), abs=1e-5) == 1.0


def test_deserialize_corrupt_bytes_raises_error(recognizer):
    with pytest.raises(ValueError, match="Invalid embedding byte length"):
        recognizer.deserialize_embedding(b"too_short")

