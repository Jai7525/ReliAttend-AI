"""
ReliAttend AI — Computer Vision Core (Detection & Quality Engine)
Independent of database and web route layers.
Uses OpenCV YuNet ONNX model for face detection and landmark localization.
Performs face-region blur and brightness quality checks with configurable thresholds.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import os
import cv2
import numpy as np


@dataclass
class QualityMetrics:
    blur_score: float  # Laplacian variance computed on face crop
    brightness_score: float  # Mean grayscale intensity of face crop
    face_width: int
    face_height: int
    detection_confidence: float
    is_blur_acceptable: bool
    is_brightness_acceptable: bool
    is_resolution_acceptable: bool


@dataclass
class FaceDetectionResult:
    face_count: int
    is_valid: bool
    error_message: Optional[str] = None
    face_bbox: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)
    face_data: Optional[np.ndarray] = None  # Full YuNet output row (15 elements)
    quality_metrics: Optional[QualityMetrics] = None


class FaceDetector:
    """
    OpenCV YuNet face detector wrapper with configurable quality verification.
    """

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.85,
        nms_threshold: float = 0.3,
        min_face_width: int = 80,
        min_face_height: int = 80,
        blur_threshold: float = 50.0,
        min_brightness: float = 40.0,
        max_brightness: float = 220.0,
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YuNet model weights not found at: {model_path}")

        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.min_face_width = min_face_width
        self.min_face_height = min_face_height
        self.blur_threshold = blur_threshold
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness

        # Default initialization input size (will be updated dynamically per image)
        self.detector = cv2.FaceDetectorYN.create(
            model=self.model_path,
            config="",
            input_size=(320, 320),
            score_threshold=self.confidence_threshold,
            nms_threshold=self.nms_threshold,
            top_k=5000,
        )

    def decode_image_bytes(self, image_bytes: bytes) -> Optional[np.ndarray]:
        """
        Safely decode raw image bytes into a BGR numpy array.
        Validates byte stream and magic numbers, rejecting truncated or corrupted data.
        """
        if not image_bytes or len(image_bytes) < 8:
            return None

        # Verify magic bytes for JPEG (FF D8 FF) or PNG (89 50 4E 47 0D 0A 1A 0A)
        is_jpeg = image_bytes[:3] == b"\xff\xd8\xff"
        is_png = image_bytes[:8] == b"\x89PNG\r\n\x1a\n"

        if not (is_jpeg or is_png):
            return None

        try:
            np_arr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is None or img.size == 0:
                return None
            return img
        except Exception:
            return None

    def compute_quality_metrics(
        self, img_bgr: np.ndarray, x: int, y: int, w: int, h: int, confidence: float
    ) -> QualityMetrics:
        """
        Computes blur, brightness, and resolution metrics specifically on the cropped face region.
        """
        img_h, img_w = img_bgr.shape[:2]

        # Clamp bounding box within image boundaries
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(img_w, x + w)
        y2 = min(img_h, y + h)

        face_crop = img_bgr[y1:y2, x1:x2]
        crop_h, crop_w = face_crop.shape[:2]

        if crop_h == 0 or crop_w == 0:
            return QualityMetrics(
                blur_score=0.0,
                brightness_score=0.0,
                face_width=0,
                face_height=0,
                detection_confidence=confidence,
                is_blur_acceptable=False,
                is_brightness_acceptable=False,
                is_resolution_acceptable=False,
            )

        face_gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)

        # 1. Blur evaluation: Laplacian variance on face region
        laplacian_var = float(cv2.Laplacian(face_gray, cv2.CV_64F).var())
        is_blur_ok = laplacian_var >= self.blur_threshold

        # 2. Brightness evaluation: Mean pixel intensity on face region
        brightness_val = float(np.mean(face_gray))
        is_bright_ok = self.min_brightness <= brightness_val <= self.max_brightness

        # 3. Resolution evaluation
        is_res_ok = crop_w >= self.min_face_width and crop_h >= self.min_face_height

        return QualityMetrics(
            blur_score=round(laplacian_var, 2),
            brightness_score=round(brightness_val, 2),
            face_width=crop_w,
            face_height=crop_h,
            detection_confidence=round(float(confidence), 4),
            is_blur_acceptable=is_blur_ok,
            is_brightness_acceptable=is_bright_ok,
            is_resolution_acceptable=is_res_ok,
        )

    def detect_and_validate(self, image_bgr: np.ndarray) -> FaceDetectionResult:
        """
        Detect faces in image, enforce exactly-one-face constraint, and evaluate quality.
        """
        if image_bgr is None or image_bgr.size == 0:
            return FaceDetectionResult(
                face_count=0,
                is_valid=False,
                error_message="Invalid or empty image provided.",
            )

        img_h, img_w = image_bgr.shape[:2]
        self.detector.setInputSize((img_w, img_h))

        _, faces = self.detector.detect(image_bgr)

        # If no faces found
        if faces is None or len(faces) == 0:
            return FaceDetectionResult(
                face_count=0,
                is_valid=False,
                error_message="No face detected in the uploaded image.",
            )

        face_count = len(faces)

        # Reject multiple faces
        if face_count > 1:
            return FaceDetectionResult(
                face_count=face_count,
                is_valid=False,
                error_message=f"Multiple faces detected ({face_count}). Exactly one face is required for reference registration.",
            )

        # Exactly 1 face detected: evaluate bounding box & quality
        face = faces[0]
        # YuNet format: [x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm, score]
        x, y, w, h = int(face[0]), int(face[1]), int(face[2]), int(face[3])
        confidence = float(face[-1])

        quality = self.compute_quality_metrics(image_bgr, x, y, w, h, confidence)

        # Check resolution
        if not quality.is_resolution_acceptable:
            return FaceDetectionResult(
                face_count=1,
                is_valid=False,
                error_message=(
                    f"Face resolution too low ({quality.face_width}x{quality.face_height}px). "
                    f"Minimum required is {self.min_face_width}x{self.min_face_height}px."
                ),
                face_bbox=(x, y, w, h),
                face_data=face,
                quality_metrics=quality,
            )

        # Check blur
        if not quality.is_blur_acceptable:
            return FaceDetectionResult(
                face_count=1,
                is_valid=False,
                error_message=(
                    f"Image is too blurry (Laplacian score: {quality.blur_score}). "
                    f"Minimum required blur threshold is {self.blur_threshold}."
                ),
                face_bbox=(x, y, w, h),
                face_data=face,
                quality_metrics=quality,
            )

        # Check brightness
        if not quality.is_brightness_acceptable:
            msg = (
                f"Face region is under-exposed/too dark (brightness: {quality.brightness_score}). Minimum required is {self.min_brightness}."
                if quality.brightness_score < self.min_brightness
                else f"Face region is over-exposed/too bright (brightness: {quality.brightness_score}). Maximum permitted is {self.max_brightness}."
            )
            return FaceDetectionResult(
                face_count=1,
                is_valid=False,
                error_message=msg,
                face_bbox=(x, y, w, h),
                face_data=face,
                quality_metrics=quality,
            )

        return FaceDetectionResult(
            face_count=1,
            is_valid=True,
            error_message=None,
            face_bbox=(x, y, w, h),
            face_data=face,
            quality_metrics=quality,
        )

