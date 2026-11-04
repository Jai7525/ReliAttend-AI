"""
ReliAttend AI — Biometric Registration & Management Service (Phase 6 & Enhancement)
Orchestrates multi-reference enrollment (up to 5 active references), consent validation,
face-region quality checks, SFace 128-d embedding extraction, secure file storage,
primary reference consistency, deletion with filesystem cleanup, and audit logging.
"""

from datetime import datetime, timezone
import json
import logging
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
    DuplicateResourceError,
    ResourceNotFoundError,
    ValidationProcessingError,
)
from app.models.academic import FacultyAssignment
from app.models.attendance import AuditLog
from app.models.student import ConsentRecord, FaceReference, Student
from app.models.user import User
from app.schemas.biometrics import (
    FaceReferenceItemResponse,
    FaceRegistrationResponse,
    FaceRegistrationStatusResponse,
    QualityMetricsResponse,
    StudentReferenceListResponse,
)

logger = logging.getLogger("reliattend.biometrics")
settings = get_settings()


class BiometricService:
    def __init__(self, db: Session):
        self.db = db
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
        - Faculty can only manage/inspect biometrics for students enrolled in classes assigned to them.
        """
        if current_user.role == "admin":
            return

        if current_user.role == "faculty":
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

    async def _process_and_extract_face(self, file: UploadFile):
        """
        Internal helper: reads upload stream safely, checks max file size,
        decodes content-based image, runs YuNet single-face detection and quality checks,
        and extracts 128-d SFace normalized embedding.
        """
        content = await file.read()
        if len(content) > settings.BIOMETRIC_MAX_UPLOAD_BYTES:
            raise BadRequestError(
                f"Uploaded image exceeds maximum permitted size of {settings.BIOMETRIC_MAX_UPLOAD_BYTES // (1024 * 1024)} MB."
            )
        if len(content) == 0:
            raise BadRequestError("Uploaded image file is empty.")

        img_bgr = self.detector.decode_image_bytes(content)
        if img_bgr is None:
            raise BadRequestError(
                "Invalid or unsupported image file. Only valid JPEG and PNG images are supported."
            )

        detection_result = self.detector.detect_and_validate(img_bgr)
        if not detection_result.is_valid:
            raise ValidationProcessingError(
                f"Face validation failed: {detection_result.error_message}"
            )

        embedding_result = self.recognizer.extract_embedding(img_bgr, detection_result.face_data)
        if not embedding_result.is_valid:
            raise ValidationProcessingError(
                f"Feature extraction failed: {embedding_result.error_message}"
            )

        return img_bgr, detection_result, embedding_result

    async def add_reference(
        self,
        student_id_str: str,
        file: UploadFile,
        current_user: User,
        pose_label: str = "frontal",
        is_primary: bool = False,
    ) -> FaceRegistrationResponse:
        """
        Adds a new reference image for a student (supporting up to 5 active references).
        Validates consent, scoped RBAC, pose label, single face, blur, brightness,
        saves image securely outside web root, and records audit trail.
        """
        # 1. Validate pose label
        if pose_label not in settings.BIOMETRIC_ALLOWED_POSES:
            raise BadRequestError(
                f"Invalid pose_label '{pose_label}'. Allowed poses: {', '.join(settings.BIOMETRIC_ALLOWED_POSES)}."
            )

        # 2. Student lookup & RBAC
        student = self.db.query(Student).filter(Student.student_id == student_id_str).first()
        if not student:
            raise ResourceNotFoundError(f"Student with ID '{student_id_str}' was not found.")

        self.verify_faculty_authorization(student, current_user)

        # 3. Active consent check
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

        # 4. Check active reference cap (maximum 5 active references)
        active_references = (
            self.db.query(FaceReference)
            .filter(
                FaceReference.student_id == student.id,
                FaceReference.status == "active",
            )
            .all()
        )
        if len(active_references) >= settings.BIOMETRIC_MAX_ACTIVE_REFERENCES:
            raise DuplicateResourceError(
                f"Student '{student_id_str}' already has the maximum permitted active reference images ({settings.BIOMETRIC_MAX_ACTIVE_REFERENCES}). "
                "Delete an outdated reference before registering a new one."
            )

        # 5. Process image & extract embedding
        img_bgr, detection_result, embedding_result = await self._process_and_extract_face(file)

        # 6. Primary reference consistency
        # If this is the student's first reference, make it primary automatically
        if not active_references:
            is_primary = True

        os.makedirs(settings.BIOMETRIC_STORAGE_DIR, exist_ok=True)
        image_filename = f"{uuid.uuid4().hex}.jpg"
        image_filepath = os.path.join(settings.BIOMETRIC_STORAGE_DIR, image_filename)

        saved_on_disk = False
        try:
            cv2.imwrite(image_filepath, img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved_on_disk = True

            # If new reference is marked primary, demote any other active reference
            if is_primary:
                for ref in active_references:
                    if ref.is_primary:
                        ref.is_primary = False

            new_ref = FaceReference(
                student_id=student.id,
                image_path=image_filepath,
                embedding=embedding_result.embedding_bytes,
                embedding_size=embedding_result.embedding_dimension,
                model_name=embedding_result.model_name,
                model_version=embedding_result.model_version,
                quality_score=detection_result.quality_metrics.detection_confidence,
                pose_label=pose_label,
                is_primary=is_primary,
                status="active",
                enrolled_by=current_user.id,
                enrolled_at=datetime.now(timezone.utc),
            )
            self.db.add(new_ref)

            student.registration_status = "enrolled"
            student.updated_at = datetime.now(timezone.utc)

            audit_entry = AuditLog(
                actor_id=current_user.id,
                action="student.face_enrolled",
                target_type="Student",
                target_id=student.id,
                details=json.dumps({
                    "student_id": student.student_id,
                    "pose_label": pose_label,
                    "is_primary": is_primary,
                    "model_name": embedding_result.model_name,
                    "model_version": embedding_result.model_version,
                    "embedding_dimension": embedding_result.embedding_dimension,
                    "detection_confidence": detection_result.quality_metrics.detection_confidence,
                    "blur_score": detection_result.quality_metrics.blur_score,
                    "brightness_score": detection_result.quality_metrics.brightness_score,
                    "superseded_previous": is_primary and len(active_references) > 0,
                    "new_active_count": len(active_references) + 1,
                }),
                timestamp=datetime.now(timezone.utc),
            )
            self.db.add(audit_entry)

            self.db.commit()
            self.db.refresh(new_ref)

        except Exception as exc:
            self.db.rollback()
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
            registration_id=new_ref.id,
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
            pose_label=new_ref.pose_label,
            is_primary=new_ref.is_primary,
            active_reference_count=len(active_references) + 1,
            model_name=embedding_result.model_name,
            embedding_dimension=embedding_result.embedding_dimension,
            created_at=new_ref.enrolled_at,
        )

    async def register_student_face(
        self,
        student_id_str: str,
        file: UploadFile,
        current_user: User,
    ) -> FaceRegistrationResponse:
        """
        Backward-compatibility endpoint: registers/updates a primary frontal reference.
        """
        # Search if student already has a primary reference or active frontal reference
        superseded_any = False
        student = self.db.query(Student).filter(Student.student_id == student_id_str).first()
        if student:
            # Check for existing primary reference
            existing_primary = (
                self.db.query(FaceReference)
                .filter(
                    FaceReference.student_id == student.id,
                    FaceReference.status == "active",
                    FaceReference.is_primary == True,
                )
                .first()
            )
            if existing_primary:
                # Mark previous primary reference as superseded
                existing_primary.status = "superseded"
                existing_primary.is_primary = False
                existing_primary.deactivated_at = datetime.now(timezone.utc)
                self.db.commit()
                superseded_any = True

        res = await self.add_reference(
            student_id_str=student_id_str,
            file=file,
            current_user=current_user,
            pose_label="frontal",
            is_primary=True,
        )

        if superseded_any:
            # Update latest audit log detail with superseded_previous = True for backward compatibility
            latest_audit = (
                self.db.query(AuditLog)
                .filter(
                    AuditLog.target_id == student.id,
                    AuditLog.action == "student.face_enrolled",
                )
                .order_by(AuditLog.id.desc())
                .first()
            )
            if latest_audit:
                d = json.loads(latest_audit.details)
                d["superseded_previous"] = True
                latest_audit.details = json.dumps(d)
                self.db.commit()

        return res

    def list_references(
        self,
        student_id_str: str,
        current_user: User,
    ) -> StudentReferenceListResponse:
        """
        Lists all active references for a student.
        Returns metadata only (ID, pose, primary flag, quality score, enrolled date).
        Zero image paths and zero embeddings exposed.
        """
        student = self.db.query(Student).filter(Student.student_id == student_id_str).first()
        if not student:
            raise ResourceNotFoundError(f"Student with ID '{student_id_str}' was not found.")

        self.verify_faculty_authorization(student, current_user)

        active_refs = (
            self.db.query(FaceReference)
            .filter(
                FaceReference.student_id == student.id,
                FaceReference.status == "active",
            )
            .order_by(FaceReference.is_primary.desc(), FaceReference.enrolled_at.desc())
            .all()
        )

        items = [
            FaceReferenceItemResponse(
                reference_id=ref.id,
                pose_label=ref.pose_label,
                is_primary=ref.is_primary,
                status=ref.status,
                quality_score=ref.quality_score,
                model_name=ref.model_name,
                embedding_dimension=ref.embedding_size,
                enrolled_at=ref.enrolled_at,
            )
            for ref in active_refs
        ]

        return StudentReferenceListResponse(
            student_id=student.student_id,
            student_name=student.name,
            active_count=len(items),
            max_allowed=settings.BIOMETRIC_MAX_ACTIVE_REFERENCES,
            references=items,
        )

    def delete_reference(
        self,
        student_id_str: str,
        reference_id: int,
        current_user: User,
    ) -> dict:
        """
        Deactivates an active reference record, unlinks its file from storage,
        promotes another active reference if the deleted reference was primary,
        and marks student pending if active count drops to 0.
        """
        student = self.db.query(Student).filter(Student.student_id == student_id_str).first()
        if not student:
            raise ResourceNotFoundError(f"Student with ID '{student_id_str}' was not found.")

        self.verify_faculty_authorization(student, current_user)

        ref = (
            self.db.query(FaceReference)
            .filter(
                FaceReference.id == reference_id,
                FaceReference.student_id == student.id,
            )
            .first()
        )
        if not ref or ref.status != "active":
            raise ResourceNotFoundError(
                f"Active face reference with ID {reference_id} was not found for student '{student_id_str}'."
            )

        file_to_delete = ref.image_path
        was_primary = ref.is_primary

        try:
            # 1. Update database status first
            ref.status = "deleted"
            ref.is_primary = False
            ref.deactivated_at = datetime.now(timezone.utc)

            # 2. Check remaining active references
            remaining_active = (
                self.db.query(FaceReference)
                .filter(
                    FaceReference.student_id == student.id,
                    FaceReference.status == "active",
                    FaceReference.id != ref.id,
                )
                .order_by(FaceReference.enrolled_at.desc())
                .all()
            )

            # If deleted reference was primary and others exist, promote the newest one
            if was_primary and remaining_active:
                remaining_active[0].is_primary = True

            # If no active references remain, set student to pending
            if not remaining_active:
                student.registration_status = "pending"
            student.updated_at = datetime.now(timezone.utc)

            # 3. Log audit event
            audit_entry = AuditLog(
                actor_id=current_user.id,
                action="student.face_reference_deleted",
                target_type="Student",
                target_id=student.id,
                details=json.dumps({
                    "student_id": student.student_id,
                    "deleted_reference_id": reference_id,
                    "was_primary": was_primary,
                    "remaining_active_count": len(remaining_active),
                }),
                timestamp=datetime.now(timezone.utc),
            )
            self.db.add(audit_entry)

            self.db.commit()

            # 4. Safe filesystem cleanup after successful DB commit
            if file_to_delete and os.path.exists(file_to_delete):
                try:
                    os.remove(file_to_delete)
                except OSError as exc:
                    logger.warning("Failed to remove deleted reference file '%s': %s", file_to_delete, exc)

        except Exception as exc:
            self.db.rollback()
            raise exc

        return {
            "success": True,
            "message": f"Reference {reference_id} deleted successfully.",
            "remaining_active_references": len(remaining_active),
            "registration_status": student.registration_status,
        }

    def get_registration_status(
        self,
        student_id_str: str,
        current_user: User,
    ) -> FaceRegistrationStatusResponse:
        """
        Returns biometric status for a student:
        enrolment status, total count, active count (up to 5), active pose angles, and model metadata.
        Zero sensitive embeddings or file paths exposed.
        """
        student = self.db.query(Student).filter(Student.student_id == student_id_str).first()
        if not student:
            raise ResourceNotFoundError(f"Student with ID '{student_id_str}' was not found.")

        self.verify_faculty_authorization(student, current_user)

        all_references = self.db.query(FaceReference).filter(FaceReference.student_id == student.id).all()
        total_count = len(all_references)
        active_refs = [r for r in all_references if r.status == "active"]
        active_count = len(active_refs)

        primary_ref = [r for r in active_refs if r.is_primary]
        primary_id = primary_ref[0].id if primary_ref else None

        active_poses = [r.pose_label for r in active_refs]

        latest_ref = None
        if active_refs:
            latest_ref = sorted(active_refs, key=lambda r: r.enrolled_at, reverse=True)[0]
        elif all_references:
            latest_ref = sorted(all_references, key=lambda r: r.enrolled_at, reverse=True)[0]

        return FaceRegistrationStatusResponse(
            student_id=student.student_id,
            student_name=student.name,
            is_enrolled=(active_count >= settings.BIOMETRIC_MIN_ACTIVE_REFERENCES and student.registration_status == "enrolled"),
            reference_count=total_count,
            active_reference_count=active_count,
            registration_status=student.registration_status,
            primary_reference_id=primary_id,
            active_poses=active_poses,
            model_name=latest_ref.model_name if latest_ref else None,
            embedding_dimension=latest_ref.embedding_size if latest_ref else None,
            latest_registration_at=latest_ref.enrolled_at if latest_ref else None,
        )
