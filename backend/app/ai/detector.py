"""
ReliAttend AI — Computer Vision Core (Detection & Quality Engine)
Independent of database and web route layers.
Uses OpenCV YuNet ONNX model for face detection and landmark localization.
Performs face-region blur and brightness quality checks with configurable thresholds.
"""

import os
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
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
    """Enrollment-specific single-face result (Phase 6)."""
    face_count: int
    is_valid: bool
    error_message: Optional[str] = None
    face_bbox: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)
    face_data: Optional[np.ndarray] = None  # Full YuNet output row (15 elements)
    quality_metrics: Optional[QualityMetrics] = None


@dataclass
class DetectedFaceBox:
    """Individual face detection in a multi-face frame (Phase 7 & 8)."""
    face_id: int
    bounding_box: Tuple[int, int, int, int]  # (x, y, width, height) in original frame pixels
    confidence: float
    landmarks: Dict[str, Tuple[int, int]]  # 5 landmarks in original frame pixels: right_eye, left_eye, nose_tip, mouth_right, mouth_left
    quality: QualityMetrics
    is_quality_acceptable: bool
    face_data: Optional[np.ndarray] = None  # Full 15-element raw row for landmark alignment in Phase 8


@dataclass
class MultiFaceDetectionResult:
    """Overall multi-face detection response for an input frame (Phase 7)."""
    face_count: int
    faces: List[DetectedFaceBox]
    frame_width: int
    frame_height: int
    inference_time_ms: float
    is_scaled: bool = False
    scale_factor: float = 1.0


class FaceDetector:
    """
    OpenCV YuNet face detector wrapper with configurable quality verification.
    Thread-safe implementation supporting single-reference validation (Phase 6)
    and multi-face detection in group frames (Phase 7).
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
        self._lock = threading.Lock()

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
        Preserved for Phase 6 enrollment backward compatibility.
        """
        if image_bgr is None or image_bgr.size == 0:
            return FaceDetectionResult(
                face_count=0,
                is_valid=False,
                error_message="Invalid or empty image provided.",
            )

        with self._lock:
            img_h, img_w = image_bgr.shape[:2]
            self.detector.setInputSize((img_w, img_h))
            self.detector.setScoreThreshold(self.confidence_threshold)
            self.detector.setNMSThreshold(self.nms_threshold)
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

    def detect_multi_faces(
        self,
        image_bgr: np.ndarray,
        score_threshold: Optional[float] = None,
        nms_threshold: Optional[float] = None,
        max_faces: int = 150,
        max_inference_dimension: int = 1920,
    ) -> MultiFaceDetectionResult:
        """
        Detect zero, one, two, or multiple faces in an image/frame (Phase 7).
        Thread-safe, transforms coordinates back to original frame space if scaled,
        and evaluates per-face landmarks and quality metrics.
        """
        import time
        start_time = time.perf_counter()

        if image_bgr is None or image_bgr.size == 0:
            return MultiFaceDetectionResult(
                face_count=0,
                faces=[],
                frame_width=0,
                frame_height=0,
                inference_time_ms=0.0,
            )

        orig_h, orig_w = image_bgr.shape[:2]
        effective_score_thresh = score_threshold if score_threshold is not None else self.confidence_threshold
        effective_nms_thresh = nms_threshold if nms_threshold is not None else self.nms_threshold

        # Aspect-ratio preserving downscale if image exceeds max inference dimension
        scale_factor = 1.0
        is_scaled = False
        det_img = image_bgr

        max_dim = max(orig_w, orig_h)
        if max_dim > max_inference_dimension and max_inference_dimension > 0:
            scale_factor = max_inference_dimension / float(max_dim)
            new_w = max(1, int(orig_w * scale_factor))
            new_h = max(1, int(orig_h * scale_factor))
            det_img = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
            is_scaled = True

        infer_h, infer_w = det_img.shape[:2]

        with self._lock:
            self.detector.setInputSize((infer_w, infer_h))
            self.detector.setScoreThreshold(effective_score_thresh)
            self.detector.setNMSThreshold(effective_nms_thresh)
            _, raw_faces = self.detector.detect(det_img)

        inference_time_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

        if raw_faces is None or len(raw_faces) == 0:
            return MultiFaceDetectionResult(
                face_count=0,
                faces=[],
                frame_width=orig_w,
                frame_height=orig_h,
                inference_time_ms=inference_time_ms,
                is_scaled=is_scaled,
                scale_factor=round(scale_factor, 4),
            )

        detected_boxes: List[DetectedFaceBox] = []
        inv_scale = 1.0 / scale_factor if is_scaled else 1.0

        # Sort raw detections by confidence descending and limit by max_faces safety cap
        sorted_faces = sorted(raw_faces, key=lambda f: float(f[-1]), reverse=True)[:max_faces]

        for idx, face in enumerate(sorted_faces, start=1):
            conf = round(float(face[-1]), 4)

            # Map coordinates back to original frame space
            if is_scaled:
                x = int(round(float(face[0]) * inv_scale))
                y = int(round(float(face[1]) * inv_scale))
                w = int(round(float(face[2]) * inv_scale))
                h = int(round(float(face[3]) * inv_scale))
                re_x, re_y = int(round(float(face[4]) * inv_scale)), int(round(float(face[5]) * inv_scale))
                le_x, le_y = int(round(float(face[6]) * inv_scale)), int(round(float(face[7]) * inv_scale))
                nt_x, nt_y = int(round(float(face[8]) * inv_scale)), int(round(float(face[9]) * inv_scale))
                rcm_x, rcm_y = int(round(float(face[10]) * inv_scale)), int(round(float(face[11]) * inv_scale))
                lcm_x, lcm_y = int(round(float(face[12]) * inv_scale)), int(round(float(face[13]) * inv_scale))
            else:
                x = int(face[0])
                y = int(face[1])
                w = int(face[2])
                h = int(face[3])
                re_x, re_y = int(face[4]), int(face[5])
                le_x, le_y = int(face[6]), int(face[7])
                nt_x, nt_y = int(face[8]), int(face[9])
                rcm_x, rcm_y = int(face[10]), int(face[11])
                lcm_x, lcm_y = int(face[12]), int(face[13])

            # Clip bbox to original image boundaries safely
            x_clipped = max(0, min(x, orig_w - 1))
            y_clipped = max(0, min(y, orig_h - 1))
            w_clipped = max(1, min(w, orig_w - x_clipped))
            h_clipped = max(1, min(h, orig_h - y_clipped))

            # Compute quality metrics on original frame's crop
            quality = self.compute_quality_metrics(
                image_bgr, x_clipped, y_clipped, w_clipped, h_clipped, conf
            )
            is_qual_ok = (
                quality.is_blur_acceptable
                and quality.is_brightness_acceptable
                and quality.is_resolution_acceptable
            )

            landmarks = {
                "right_eye": (re_x, re_y),
                "left_eye": (le_x, le_y),
                "nose_tip": (nt_x, nt_y),
                "mouth_right": (rcm_x, rcm_y),
                "mouth_left": (lcm_x, lcm_y),
            }

            # In scaled mode, also create an unscaled face_data row for recognition alignment
            if is_scaled:
                orig_face_data = np.array([
                    x, y, w, h,
                    re_x, re_y, le_x, le_y, nt_x, nt_y,
                    rcm_x, rcm_y, lcm_x, lcm_y,
                    conf
                ], dtype=np.float32)
            else:
                orig_face_data = face.copy()

            detected_boxes.append(
                DetectedFaceBox(
                    face_id=idx,
                    bounding_box=(x, y, w, h),
                    confidence=conf,
                    landmarks=landmarks,
                    quality=quality,
                    is_quality_acceptable=is_qual_ok,
                    face_data=orig_face_data,
                )
            )

        return MultiFaceDetectionResult(
            face_count=len(detected_boxes),
            faces=detected_boxes,
            frame_width=orig_w,
            frame_height=orig_h,
            inference_time_ms=inference_time_ms,
            is_scaled=is_scaled,
            scale_factor=round(scale_factor, 4),
        )

