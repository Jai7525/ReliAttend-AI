"""
ReliAttend AI — Multi-Face Recognition Service (Phase 8)
Orchestrates classroom frame detection, face alignment, embedding extraction,
and 1:N gallery matching scoped strictly to the requested class section.
Safeguards privacy: zero filesystem persistence, zero embedding leakage.
"""

import logging
import os
import time
from typing import List, Optional, Tuple
import cv2
from fastapi import UploadFile
import numpy as np
from sqlalchemy.orm import Session

from app.ai.detector import FaceDetector, MultiFaceDetectionResult
from app.ai.matcher import CandidateMatch, FaceMatchResult, GalleryItem, VectorMatcher
from app.ai.recognition import FaceRecognizer
from app.config import get_settings
from app.core.exceptions import (
    AuthorizationForbiddenError,
    BadRequestError,
    ResourceNotFoundError,
)
from app.models.academic import ClassSection, FacultyAssignment
from app.models.student import ConsentRecord, FaceReference, Student
from app.models.user import User
from app.schemas.detection import (
    BoundingBoxSchema,
    FaceQualitySchema,
    FrameMetadataSchema,
)
from app.schemas.recognition import (
    FrameRecognitionData,
    FrameRecognitionResponse,
    MatchCandidateSchema,
    RecognitionSummarySchema,
    RecognizedFaceItemSchema,
)

logger = logging.getLogger("reliattend.recognition")
settings = get_settings()


class RecognitionService:
    """
    Stateless 1:N multi-face recognition service for classroom attendance frames.
    """

    def __init__(
        self,
        db: Session,
        detector: Optional[FaceDetector] = None,
        recognizer: Optional[FaceRecognizer] = None,
    ):
        self.db = db

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

        if recognizer is not None:
            self.recognizer = recognizer
        else:
            sface_path = os.path.join(
                settings.BIOMETRIC_MODELS_DIR, "face_recognition_sface_2021dec.onnx"
            )
            self.recognizer = FaceRecognizer(model_path=sface_path)

    def _verify_section_authorization(self, class_section_id: int, user: User) -> ClassSection:
        """Enforces that only Admin or assigned Faculty can access the section gallery."""
        section = self.db.query(ClassSection).filter_by(id=class_section_id).first()
        if not section:
            raise ResourceNotFoundError(f"Class section ID {class_section_id} not found.")

        if user.role == "admin":
            return section

        # Check faculty assignment
        assigned = (
            self.db.query(FacultyAssignment)
            .filter_by(user_id=user.id, class_section_id=class_section_id, is_active=True)
            .first()
        )
        if not assigned:
            raise AuthorizationForbiddenError(
                f"Faculty '{user.username}' is not assigned to class section '{section.name}'."
            )
        return section

    def _load_section_gallery(self, class_section_id: int) -> Tuple[List[GalleryItem], int]:
        """
        Loads all active reference embeddings for students enrolled in the class section.
        Strict DPDP rule: Excludes students without active consent.
        """
        # Query enrolled, active students in this section who have active consent
        students = (
            self.db.query(Student)
            .filter_by(class_section_id=class_section_id, is_active=True)
            .all()
        )

        gallery: List[GalleryItem] = []
        eligible_students_count = 0

        for student in students:
            # Check consent
            active_consent = (
                self.db.query(ConsentRecord)
                .filter_by(student_id=student.id, is_active=True)
                .first()
            )
            if not active_consent:
                continue

            # Query active face references
            active_refs = (
                self.db.query(FaceReference)
                .filter_by(student_id=student.id, status="active")
                .all()
            )

            if not active_refs:
                continue

            eligible_students_count += 1
            for ref in active_refs:
                try:
                    emb = FaceRecognizer.deserialize_embedding(ref.embedding)
                    gallery.append(
                        GalleryItem(
                            reference_id=ref.id,
                            student_id=student.id,
                            student_str_id=student.student_id,
                            student_name=student.name,
                            pose_label=ref.pose_label or "frontal",
                            is_primary=ref.is_primary,
                            embedding=emb,
                        )
                    )
                except Exception as e:
                    logger.warning(
                        "Skipping corrupted embedding for reference_id=%d, student_id=%s: %s",
                        ref.id,
                        student.student_id,
                        str(e),
                    )

        return gallery, eligible_students_count

    async def recognize_frame_for_section(
        self,
        class_section_id: int,
        file: UploadFile,
        current_user: User,
        similarity_threshold: Optional[float] = None,
        ambiguity_margin: Optional[float] = None,
    ) -> FrameRecognitionResponse:
        """
        Detects faces in frame and performs 1:N recognition against the section gallery.
        """
        start_time = time.perf_counter()

        # 1. Authorize user for section
        section = self._verify_section_authorization(class_section_id, current_user)

        # 2. Validate input file
        if file.content_type not in settings.BIOMETRIC_ALLOWED_MIME_TYPES:
            raise BadRequestError(
                f"Unsupported file type '{file.content_type}'. Allowed types: {', '.join(settings.BIOMETRIC_ALLOWED_MIME_TYPES)}"
            )

        image_bytes = await file.read()
        file_size = len(image_bytes)

        if file_size == 0:
            raise BadRequestError("Uploaded file is empty.")

        if file_size > settings.BIOMETRIC_MAX_UPLOAD_BYTES:
            max_mb = settings.BIOMETRIC_MAX_UPLOAD_BYTES // (1024 * 1024)
            raise BadRequestError(
                f"Image file size ({file_size / (1024 * 1024):.2f} MB) exceeds maximum allowed limit ({max_mb} MB)."
            )

        # 3. Decode in-memory (Zero disk storage)
        image_bgr = self.detector.decode_image_bytes(image_bytes)
        if image_bgr is None:
            raise BadRequestError("Failed to decode image. Ensure valid, uncorrupted JPEG or PNG.")

        orig_h, orig_w = image_bgr.shape[:2]

        if orig_w > settings.DETECTION_MAX_FRAME_WIDTH or orig_h > settings.DETECTION_MAX_FRAME_HEIGHT:
            raise BadRequestError(
                f"Image dimensions ({orig_w}x{orig_h}) exceed limit ({settings.DETECTION_MAX_FRAME_WIDTH}x{settings.DETECTION_MAX_FRAME_HEIGHT})."
            )

        # 4. Load gallery for section
        gallery, eligible_students_count = self._load_section_gallery(class_section_id)

        # 5. Detect faces in frame
        detection_res: MultiFaceDetectionResult = self.detector.detect_multi_faces(
            image_bgr=image_bgr,
            score_threshold=settings.DETECTION_DEFAULT_CONFIDENCE_THRESHOLD,
            nms_threshold=settings.DETECTION_DEFAULT_NMS_THRESHOLD,
            max_faces=settings.DETECTION_MAX_FACES_PER_FRAME,
            max_inference_dimension=settings.DETECTION_INFERENCE_MAX_DIMENSION,
        )

        # 6. Extract SFace embeddings for each detected face
        live_embeddings: List[Tuple[int, np.ndarray]] = []
        recognized_face_items: List[RecognizedFaceItemSchema] = []

        effective_sim_thresh = (
            similarity_threshold
            if similarity_threshold is not None
            else settings.RECOGNITION_EXPERIMENTAL_MATCH_THRESHOLD
        )
        effective_margin = (
            ambiguity_margin
            if ambiguity_margin is not None
            else settings.RECOGNITION_EXPERIMENTAL_AMBIGUITY_MARGIN
        )

        for det_face in detection_res.faces:
            bbox = BoundingBoxSchema(
                x=det_face.bounding_box[0],
                y=det_face.bounding_box[1],
                width=det_face.bounding_box[2],
                height=det_face.bounding_box[3],
            )
            quality_schema = FaceQualitySchema(
                blur_score=det_face.quality.blur_score,
                brightness_score=det_face.quality.brightness_score,
                face_width=det_face.quality.face_width,
                face_height=det_face.quality.face_height,
                is_blur_acceptable=det_face.quality.is_blur_acceptable,
                is_brightness_acceptable=det_face.quality.is_brightness_acceptable,
                is_resolution_acceptable=det_face.quality.is_resolution_acceptable,
                is_acceptable=det_face.is_quality_acceptable,
            )

            # Check if face is too small or degenerate for landmark extraction
            if det_face.face_data is None or bbox.width < 20 or bbox.height < 20:
                recognized_face_items.append(
                    RecognizedFaceItemSchema(
                        face_id=det_face.face_id,
                        bounding_box=bbox,
                        detection_confidence=det_face.confidence,
                        quality=quality_schema,
                        match_status="unrecognized",
                        top_match=None,
                        top_candidates=[],
                        margin_to_second=0.0,
                        reason="Face resolution too small for reliable feature extraction.",
                    )
                )
                continue

            try:
                emb_res = self.recognizer.extract_embedding(image_bgr, det_face.face_data)
                if not emb_res.is_valid:
                    recognized_face_items.append(
                        RecognizedFaceItemSchema(
                            face_id=det_face.face_id,
                            bounding_box=bbox,
                            detection_confidence=det_face.confidence,
                            quality=quality_schema,
                            match_status="unrecognized",
                            top_match=None,
                            top_candidates=[],
                            margin_to_second=0.0,
                            reason=emb_res.error_message or "Feature extraction failed.",
                        )
                    )
                else:
                    live_emb = FaceRecognizer.deserialize_embedding(emb_res.embedding_bytes)
                    live_embeddings.append((det_face.face_id, live_emb))
            except Exception as exc:
                logger.warning("Feature extraction error on face_id=%d: %s", det_face.face_id, str(exc))
                recognized_face_items.append(
                    RecognizedFaceItemSchema(
                        face_id=det_face.face_id,
                        bounding_box=bbox,
                        detection_confidence=det_face.confidence,
                        quality=quality_schema,
                        match_status="unrecognized",
                        top_match=None,
                        top_candidates=[],
                        margin_to_second=0.0,
                        reason=f"Feature extraction exception: {str(exc)}",
                    )
                )

        # 7. Perform 1:N matching across gallery
        if live_embeddings:
            matcher = VectorMatcher(
                gallery=gallery,
                similarity_threshold=effective_sim_thresh,
                ambiguity_margin=effective_margin,
                top_k=settings.RECOGNITION_TOP_K,
            )
            match_results = matcher.match_embeddings(live_embeddings)
            match_map = {r.face_id: r for r in match_results}

            for det_face in detection_res.faces:
                if det_face.face_id in match_map:
                    m = match_map[det_face.face_id]
                    bbox = BoundingBoxSchema(
                        x=det_face.bounding_box[0],
                        y=det_face.bounding_box[1],
                        width=det_face.bounding_box[2],
                        height=det_face.bounding_box[3],
                    )
                    quality_schema = FaceQualitySchema(
                        blur_score=det_face.quality.blur_score,
                        brightness_score=det_face.quality.brightness_score,
                        face_width=det_face.quality.face_width,
                        face_height=det_face.quality.face_height,
                        is_blur_acceptable=det_face.quality.is_blur_acceptable,
                        is_brightness_acceptable=det_face.quality.is_brightness_acceptable,
                        is_resolution_acceptable=det_face.quality.is_resolution_acceptable,
                        is_acceptable=det_face.is_quality_acceptable,
                    )

                    top_m_schema = (
                        MatchCandidateSchema(
                            student_id=m.top_match.student_str_id,
                            student_name=m.top_match.student_name,
                            similarity_score=m.top_match.similarity_score,
                            matched_pose=m.top_match.matched_pose,
                            is_primary=m.top_match.is_primary,
                        )
                        if m.top_match
                        else None
                    )

                    top_cands_schema = [
                        MatchCandidateSchema(
                            student_id=c.student_str_id,
                            student_name=c.student_name,
                            similarity_score=c.similarity_score,
                            matched_pose=c.matched_pose,
                            is_primary=c.is_primary,
                        )
                        for c in m.top_candidates
                    ]

                    recognized_face_items.append(
                        RecognizedFaceItemSchema(
                            face_id=det_face.face_id,
                            bounding_box=bbox,
                            detection_confidence=det_face.confidence,
                            quality=quality_schema,
                            match_status=m.match_status,
                            top_match=top_m_schema,
                            top_candidates=top_cands_schema,
                            margin_to_second=m.margin_to_second,
                            reason=m.reason,
                        )
                    )

        # Sort output faces by face_id
        recognized_face_items.sort(key=lambda item: item.face_id)

        # 8. Calculate counts (Requirement 8: faces_recognized ONLY counts 'matched')
        count_matched = sum(1 for f in recognized_face_items if f.match_status == "matched")
        count_uncertain = sum(1 for f in recognized_face_items if f.match_status == "uncertain")
        count_unrecognized = sum(1 for f in recognized_face_items if f.match_status == "unrecognized")

        total_elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

        summary = RecognitionSummarySchema(
            section_id=section.id,
            section_name=section.name,
            faces_detected=len(recognized_face_items),
            faces_recognized=count_matched,
            faces_uncertain=count_uncertain,
            faces_unrecognized=count_unrecognized,
            gallery_students_count=eligible_students_count,
            gallery_references_count=len(gallery),
        )

        metadata = FrameMetadataSchema(
            width=orig_w,
            height=orig_h,
            channels=image_bgr.shape[2] if len(image_bgr.shape) > 2 else 1,
            inference_time_ms=total_elapsed_ms,
            is_scaled=detection_res.is_scaled,
            scale_factor=detection_res.scale_factor,
        )

        logger.info(
            "Recognition on section %d completed in %.2f ms: %d detected, %d matched, %d uncertain, %d unrecognized",
            section.id,
            total_elapsed_ms,
            len(recognized_face_items),
            count_matched,
            count_uncertain,
            count_unrecognized,
        )

        return FrameRecognitionResponse(
            success=True,
            message="Multi-face recognition completed successfully.",
            data=FrameRecognitionData(
                summary=summary,
                frame_metadata=metadata,
                faces=recognized_face_items,
            ),
        )
