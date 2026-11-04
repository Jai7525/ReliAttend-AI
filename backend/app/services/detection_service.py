"""
ReliAttend AI — Multi-Face Detection Service (Phase 7)
Orchestrates multi-face detection in classroom group frames.
Enforces security boundaries, memory-only byte decoding, coordinate normalization,
thread-safe detector invocation, and zero filesystem persistence.
"""

import logging
import os
from typing import Optional
from fastapi import UploadFile

from app.ai.detector import FaceDetector, MultiFaceDetectionResult
from app.config import get_settings
from app.core.exceptions import BadRequestError
from app.models.user import User
from app.schemas.detection import (
    BoundingBoxSchema,
    DetectedFaceSchema,
    FaceQualitySchema,
    FacialLandmarksSchema,
    FrameMetadataSchema,
    MultiFaceDetectionData,
    MultiFaceDetectionResponse,
)

logger = logging.getLogger("reliattend.detection")
settings = get_settings()


class DetectionService:
    """
    Stateless multi-face detection service.
    Processes group images in-memory and returns face bounding boxes,
    landmarks, detector confidences, and quality metrics.
    """

    def __init__(self, detector: Optional[FaceDetector] = None):
        if detector is not None:
            self.detector = detector
        else:
            yunet_path = os.path.join(
                settings.BIOMETRIC_MODELS_DIR, "face_detection_yunet_2023mar.onnx"
            )
            self.detector = FaceDetector(
                model_path=yunet_path,
                confidence_threshold=settings.DETECTION_DEFAULT_CONFIDENCE_THRESHOLD,
                nms_threshold=settings.DETECTION_DEFAULT_NMS_THRESHOLD,
                min_face_width=settings.BIOMETRIC_MIN_FACE_WIDTH,
                min_face_height=settings.BIOMETRIC_MIN_FACE_HEIGHT,
                blur_threshold=settings.BIOMETRIC_BLUR_THRESHOLD,
                min_brightness=settings.BIOMETRIC_MIN_BRIGHTNESS,
                max_brightness=settings.BIOMETRIC_MAX_BRIGHTNESS,
            )

    async def detect_faces_in_frame(
        self,
        file: UploadFile,
        current_user: User,
        confidence_threshold: Optional[float] = None,
        nms_threshold: Optional[float] = None,
    ) -> MultiFaceDetectionResponse:
        """
        Validate incoming file, decode in-memory, run multi-face detector,
        and format response. Zero disk persistence.
        """
        # 1. Validate MIME Type
        if file.content_type not in settings.BIOMETRIC_ALLOWED_MIME_TYPES:
            raise BadRequestError(
                f"Unsupported file type '{file.content_type}'. Allowed types: {', '.join(settings.BIOMETRIC_ALLOWED_MIME_TYPES)}"
            )

        # 2. Read bytes safely with size check
        image_bytes = await file.read()
        file_size = len(image_bytes)

        if file_size == 0:
            raise BadRequestError("Uploaded file is empty.")

        if file_size > settings.BIOMETRIC_MAX_UPLOAD_BYTES:
            max_mb = settings.BIOMETRIC_MAX_UPLOAD_BYTES // (1024 * 1024)
            raise BadRequestError(
                f"Image file size ({file_size / (1024 * 1024):.2f} MB) exceeds maximum allowed limit ({max_mb} MB)."
            )

        # 3. Decode image bytes completely in-memory
        image_bgr = self.detector.decode_image_bytes(image_bytes)
        if image_bgr is None:
            raise BadRequestError(
                "Failed to decode image. Ensure the file is a valid, uncorrupted JPEG or PNG image."
            )

        orig_h, orig_w = image_bgr.shape[:2]

        # 4. Check dimension limits
        if (
            orig_w > settings.DETECTION_MAX_FRAME_WIDTH
            or orig_h > settings.DETECTION_MAX_FRAME_HEIGHT
        ):
            raise BadRequestError(
                f"Image dimensions ({orig_w}x{orig_h}) exceed maximum permitted size "
                f"({settings.DETECTION_MAX_FRAME_WIDTH}x{settings.DETECTION_MAX_FRAME_HEIGHT})."
            )

        # 5. Validate optional threshold parameters if provided
        if confidence_threshold is not None and not (0.01 <= confidence_threshold <= 1.0):
            raise BadRequestError("Confidence threshold must be between 0.01 and 1.0.")

        if nms_threshold is not None and not (0.01 <= nms_threshold <= 1.0):
            raise BadRequestError("NMS threshold must be between 0.01 and 1.0.")

        # 6. Execute multi-face detection (Thread-safe)
        result: MultiFaceDetectionResult = self.detector.detect_multi_faces(
            image_bgr=image_bgr,
            score_threshold=confidence_threshold,
            nms_threshold=nms_threshold,
            max_faces=settings.DETECTION_MAX_FACES_PER_FRAME,
            max_inference_dimension=settings.DETECTION_INFERENCE_MAX_DIMENSION,
        )

        logger.info(
            "Multi-face detection completed: detected %d faces in %dx%d frame in %.2f ms (user_id=%d)",
            result.face_count,
            result.frame_width,
            result.frame_height,
            result.inference_time_ms,
            current_user.id,
        )

        # 7. Build response payload (Absolute original pixel coordinates)
        face_items = [
            DetectedFaceSchema(
                face_id=f.face_id,
                bounding_box=BoundingBoxSchema(
                    x=f.bounding_box[0],
                    y=f.bounding_box[1],
                    width=f.bounding_box[2],
                    height=f.bounding_box[3],
                ),
                confidence=f.confidence,
                landmarks=FacialLandmarksSchema(
                    right_eye=f.landmarks["right_eye"],
                    left_eye=f.landmarks["left_eye"],
                    nose_tip=f.landmarks["nose_tip"],
                    mouth_right=f.landmarks["mouth_right"],
                    mouth_left=f.landmarks["mouth_left"],
                ),
                quality=FaceQualitySchema(
                    blur_score=f.quality.blur_score,
                    brightness_score=f.quality.brightness_score,
                    face_width=f.quality.face_width,
                    face_height=f.quality.face_height,
                    is_blur_acceptable=f.quality.is_blur_acceptable,
                    is_brightness_acceptable=f.quality.is_brightness_acceptable,
                    is_resolution_acceptable=f.quality.is_resolution_acceptable,
                    is_acceptable=f.is_quality_acceptable,
                ),
            )
            for f in result.faces
        ]

        data = MultiFaceDetectionData(
            face_count=result.face_count,
            frame_metadata=FrameMetadataSchema(
                width=result.frame_width,
                height=result.frame_height,
                channels=image_bgr.shape[2] if len(image_bgr.shape) > 2 else 1,
                inference_time_ms=result.inference_time_ms,
                is_scaled=result.is_scaled,
                scale_factor=result.scale_factor,
            ),
            faces=face_items,
        )

        return MultiFaceDetectionResponse(
            success=True,
            message="Detection completed successfully.",
            data=data,
        )

