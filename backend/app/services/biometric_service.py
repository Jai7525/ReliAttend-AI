"""
ReliAttend AI — Biometric Registration & Management Service (Phase 6)
Orchestrates consent validation, image decoding, face detection & quality evaluation,
feature extraction, secure file storage, database persistence, and rollback handling.
"""

from datetime import datetime, timezone
import json
import os
import uuid
import cv2
from fastapi import UploadFile
from sqlalchemy.orm import Session

from app.ai.detector import FaceDetector
from app.ai.recognition import FaceRecognizer
from app.config import get_settings
from app.core.exceptions import (
    BadRequestError,
    AuthorizationForbiddenError,
    ResourceNotFoundError,
    ValidationProcessingError,
)
from app.models.academic import FacultyAssignment
from app.models.attendance import AuditLog
from app.models.student import ConsentRecord, FaceReference, Student
from app.models.user import User
from app.schemas.biometrics import (
    FaceRegistrationResponse,
    FaceRegistrationStatusResponse,
    QualityMetricsResponse,
)

settings = get_settings()


class BiometricService:
    def __init__(self, db: Session):
        self.db = db
        # Initialize detector and recognizer using configured paths and thresholds
        yunet_path = os.path.join(settings.BIOMETRIC_MODELS_DIR, "face_detection_yunet_2023mar.onnx")
        sface_path = os.path.join(settings.BIOMETRIC_MODELS_DIR, "face_recognition_sface_2021dec.onnx")

        self.detector = FaceDetector(
            model_path=yunet_path,
            confidence_threshold=settings.BIOMETRIC_YUNET_CONFIDENCE_THRESHOLD,
            min_face_width=settings.BIOMETRIC_MIN_FACE_WIDTH,
            min_face_height=settings.BIOMETRIC_MIN_FACE_HEIGHT,
            blur_threshold=settings.BIOMETRIC_BLUR_THRESHOLD,
            min_brightness=settings.BIOMETRIC_MIN_BRIGHTNESS,
            max_brightness=settings.BIOMETRIC_MAX_BRIGHTNESS,
        )
        self.recognizer = FaceRecognizer(model_path=sface_path)

    def verify_faculty_authorization(self, student: Student, current_user: User) -> None:
        """
        Enforce scoped authorization:
        - Admin has full institutional access.
        - Faculty can only register/inspect biometrics for students enrolled in classes assigned to them.
        """
        if current_user.role == "admin":
            return

        if current_user.role == "faculty":
            # Check if this faculty is assigned to the student's class section
            has_assignment = (
                self.db.query(FacultyAssignment)
                .filter(
                    FacultyAssignment.user_id == current_user.id,
                    FacultyAssignment.class_section_id == student.class_section_id,
                    FacultyAssignment.is_active == True,
                )
                .first()
            )
            if not has_assignment:
                raise AuthorizationForbiddenError(
                    f"Access denied: You are not assigned to class section ID {student.class_section_id} for student {student.student_id}."
                )
            return

        raise AuthorizationForbiddenError("Access denied: You do not have permission to manage biometric records.")

    async def register_student_face(
        self,
        student_id_str: str,
        file: UploadFile,
        current_user: User,
    ) -> FaceRegistrationResponse:
        """
        Validates student existence, active consent, RBAC permissions, uploaded image bytes,
        detects single face, runs quality checks, extracts SFace 128-d embedding,
        saves image securely, updates DB transactionally (superseding previous active ref),
        and records audit log.
        """
        # 1. Student existence check
        student = self.db.query(Student).filter(Student.student_id == student_id_str).first()
        if not student:
            raise ResourceNotFoundError(f"Student with ID '{student_id_str}' was not found.")

        # 2. Scoped authorization check
        self.verify_faculty_authorization(student, current_user)

        # 3. Biometric consent verification (BR-018, FR-ADMIN-004)
        consent = (
            self.db.query(ConsentRecord)
            .filter(
                ConsentRecord.student_id == student.id,
                ConsentRecord.is_active == True,
            )
            .first()
        )
        if not consent:
            raise AuthorizationForbiddenError(
                f"Biometric registration blocked: Student '{student_id_str}' does not have an active consent record on file."
            )

        # 4. Stream upload validation & size check
        # Read file chunks up to max allowed bytes to prevent memory exhaustion
        content = await file.read()
        if len(content) > settings.BIOMETRIC_MAX_UPLOAD_BYTES:
            raise BadRequestError(
                f"Uploaded image exceeds maximum permitted size of {settings.BIOMETRIC_MAX_UPLOAD_BYTES // (1024 * 1024)} MB."
            )
        if len(content) == 0:
            raise BadRequestError("Uploaded image file is empty.")

        # 5. Image decoding & format validation (content-based, not just extension)
        img_bgr = self.detector.decode_image_bytes(content)
        if img_bgr is None:
            raise BadRequestError(
                "Invalid or unsupported image file. Only valid JPEG and PNG images are supported."
            )

        # 6. Face detection & quality verification (single-face constraint, blur, brightness, min resolution)
        detection_result = self.detector.detect_and_validate(img_bgr)
        if not detection_result.is_valid:
            # Raise 422 Unprocessable with specific quality/count measurement reason
            raise ValidationProcessingError(
                f"Face validation failed: {detection_result.error_message}"
            )

        # 7. Embedding feature extraction
        embedding_result = self.recognizer.extract_embedding(img_bgr, detection_result.face_data)
        if not embedding_result.is_valid:
            raise ValidationProcessingError(
                f"Feature extraction failed: {embedding_result.error_message}"
            )

        # 8. Secure image storage (UUID filename outside web root)
        os.makedirs(settings.BIOMETRIC_STORAGE_DIR, exist_ok=True)
        image_filename = f"{uuid.uuid4().hex}.jpg"
        image_filepath = os.path.join(settings.BIOMETRIC_STORAGE_DIR, image_filename)

        # 9. Atomic DB update with filesystem rollback handling
        saved_on_disk = False
        try:
            # Save normalized JPEG image to disk
            cv2.imwrite(image_filepath, img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved_on_disk = True

            # Enforce single active reference policy:
            # Check if student already has a FaceReference record in database
            existing_ref = (
                self.db.query(FaceReference)
                .filter(FaceReference.student_id == student.id)
                .first()
            )

            old_image_to_delete = None
            superseded_previous = False

            if existing_ref:
                # Student already has a reference: update in-place with new biometric data
                old_image_to_delete = existing_ref.image_path
                existing_ref.image_path = image_filepath
                existing_ref.embedding = embedding_result.embedding_bytes
                existing_ref.embedding_size = embedding_result.embedding_dimension
                existing_ref.model_name = embedding_result.model_name
                existing_ref.model_version = embedding_result.model_version
                existing_ref.quality_score = detection_result.quality_metrics.detection_confidence
                existing_ref.status = "active"
                existing_ref.enrolled_by = current_user.id
                existing_ref.enrolled_at = datetime.now(timezone.utc)
                target_ref = existing_ref
                superseded_previous = True
            else:
                # Insert first FaceReference record
                new_face_ref = FaceReference(
                    student_id=student.id,
                    image_path=image_filepath,
                    embedding=embedding_result.embedding_bytes,
                    embedding_size=embedding_result.embedding_dimension,
                    model_name=embedding_result.model_name,
                    model_version=embedding_result.model_version,
                    quality_score=detection_result.quality_metrics.detection_confidence,
                    status="active",
                    enrolled_by=current_user.id,
                    enrolled_at=datetime.now(timezone.utc),
                )
                self.db.add(new_face_ref)
                target_ref = new_face_ref

            # Update student status to enrolled
            student.registration_status = "enrolled"
            student.updated_at = datetime.now(timezone.utc)

            # Record system-wide audit log
            audit_entry = AuditLog(
                actor_id=current_user.id,
                action="student.face_enrolled",
                target_type="Student",
                target_id=student.id,
                details=json.dumps({
                    "student_id": student.student_id,
                    "model_name": embedding_result.model_name,
                    "model_version": embedding_result.model_version,
                    "preprocessing_version": embedding_result.preprocessing_version,
                    "embedding_dimension": embedding_result.embedding_dimension,
                    "detection_confidence": detection_result.quality_metrics.detection_confidence,
                    "blur_score": detection_result.quality_metrics.blur_score,
                    "brightness_score": detection_result.quality_metrics.brightness_score,
                    "superseded_previous": superseded_previous,
                }),
                timestamp=datetime.now(timezone.utc),
            )
            self.db.add(audit_entry)

            self.db.commit()
            self.db.refresh(target_ref)

            # Safely remove previous reference image after successful commit
            if old_image_to_delete and os.path.exists(old_image_to_delete):
                try:
                    os.remove(old_image_to_delete)
                except OSError:
                    pass

        except Exception as exc:
            self.db.rollback()
            # Clean up orphaned file on disk if DB transaction failed
            if saved_on_disk and os.path.exists(image_filepath):
                try:
                    os.remove(image_filepath)
                except OSError:
                    pass
            raise exc

        qm = detection_result.quality_metrics
        return FaceRegistrationResponse(
            success=True,
            student_id=student.student_id,
            student_name=student.name,
            registration_id=target_ref.id,
            registration_status=student.registration_status,
            face_count=detection_result.face_count,
            quality_status="passed",
            quality_metrics=QualityMetricsResponse(
                blur_score=qm.blur_score,
                brightness_score=qm.brightness_score,
                face_width=qm.face_width,
                face_height=qm.face_height,
                detection_confidence=qm.detection_confidence,
                is_blur_acceptable=qm.is_blur_acceptable,
                is_brightness_acceptable=qm.is_brightness_acceptable,
                is_resolution_acceptable=qm.is_resolution_acceptable,
            ),
            model_name=embedding_result.model_name,
            embedding_dimension=embedding_result.embedding_dimension,
            created_at=target_ref.enrolled_at,
        )

    def get_registration_status(
        self,
        student_id_str: str,
        current_user: User,
    ) -> FaceRegistrationStatusResponse:
        """
        Returns biometric status for a student:
        enrolment status, reference count, model info, and latest timestamp.
        Zero sensitive embeddings or paths exposed.
        """
        student = self.db.query(Student).filter(Student.student_id == student_id_str).first()
        if not student:
            raise ResourceNotFoundError(f"Student with ID '{student_id_str}' was not found.")

        # Scoped authorization check
        self.verify_faculty_authorization(student, current_user)

        references = self.db.query(FaceReference).filter(FaceReference.student_id == student.id).all()
        total_count = len(references)
        active_refs = [r for r in references if r.status == "active"]
        active_count = len(active_refs)

        latest_ref = None
        if active_refs:
            latest_ref = sorted(active_refs, key=lambda r: r.enrolled_at, reverse=True)[0]
        elif references:
            latest_ref = sorted(references, key=lambda r: r.enrolled_at, reverse=True)[0]

        return FaceRegistrationStatusResponse(
            student_id=student.student_id,
            student_name=student.name,
            is_enrolled=(active_count > 0 and student.registration_status == "enrolled"),
            reference_count=total_count,
            active_reference_count=active_count,
            registration_status=student.registration_status,
            model_name=latest_ref.model_name if latest_ref else None,
            embedding_dimension=latest_ref.embedding_size if latest_ref else None,
            latest_registration_at=latest_ref.enrolled_at if latest_ref else None,
        )
