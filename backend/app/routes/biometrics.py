"""
ReliAttend AI — Biometric & Face Registration Routes (Phase 6 & Enhancement)
Exposes endpoints for reference face registration, incremental multi-reference upload,
reference listing, and reference deletion.
"""

from typing import Optional
from fastapi import APIRouter, Depends, File, Form, Path, UploadFile, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.user import User
from app.routes.deps import require_faculty
from app.schemas.biometrics import (
    FaceRegistrationResponse,
    FaceRegistrationStatusResponse,
    StudentReferenceListResponse,
)
from app.services.biometric_service import BiometricService

router = APIRouter(prefix="/biometrics", tags=["Biometric Management"])


@router.post(
    "/students/{student_id}/register-face",
    response_model=FaceRegistrationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register student primary reference face (Backward-compatible)",
    description=(
        "Registers or replaces a primary frontal reference face image. "
        "Preserved for backward-compatibility with Phase 6 clients."
    ),
)
async def register_student_face(
    student_id: str = Path(..., description="Student institutional ID e.g. 'STU2026001'"),
    file: UploadFile = File(..., description="JPEG or PNG face image"),
    current_user: User = Depends(require_faculty),
    db: Session = Depends(get_db),
):
    service = BiometricService(db)
    return await service.register_student_face(
        student_id_str=student_id,
        file=file,
        current_user=current_user,
    )


@router.post(
    "/students/{student_id}/references",
    response_model=FaceRegistrationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Add an incremental reference face image (up to 5 active)",
    description=(
        "Adds an incremental reference image with a specific pose/angle label "
        "(frontal, slight_left, slight_right, slight_up, slight_down, custom). "
        "Enforces the maximum cap of 5 active references per student."
    ),
)
async def add_student_reference(
    student_id: str = Path(..., description="Student institutional ID e.g. 'STU2026001'"),
    file: UploadFile = File(..., description="JPEG or PNG face image"),
    pose_label: str = Form("frontal", description="Angle/pose label"),
    is_primary: bool = Form(False, description="Set as primary reference face"),
    current_user: User = Depends(require_faculty),
    db: Session = Depends(get_db),
):
    service = BiometricService(db)
    return await service.add_reference(
        student_id_str=student_id,
        file=file,
        current_user=current_user,
        pose_label=pose_label,
        is_primary=is_primary,
    )


@router.get(
    "/students/{student_id}/references",
    response_model=StudentReferenceListResponse,
    status_code=status.HTTP_200_OK,
    summary="List all active reference images for a student",
    description=(
        "Retrieves active reference metadata including pose labels, primary indicator, "
        "and quality scores. Zero internal file paths and zero raw embeddings exposed."
    ),
)
def list_student_references(
    student_id: str = Path(..., description="Student institutional ID e.g. 'STU2026001'"),
    current_user: User = Depends(require_faculty),
    db: Session = Depends(get_db),
):
    service = BiometricService(db)
    return service.list_references(
        student_id_str=student_id,
        current_user=current_user,
    )


@router.delete(
    "/students/{student_id}/references/{reference_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete / Deactivate a specific face reference image",
    description=(
        "Safely deactivates an active reference record and unlinks its image file from storage. "
        "If the deleted reference was primary, automatically promotes another active reference. "
        "If all references are deleted, reverts student registration status to pending."
    ),
)
def delete_student_reference(
    student_id: str = Path(..., description="Student institutional ID e.g. 'STU2026001'"),
    reference_id: int = Path(..., description="Reference ID to delete"),
    current_user: User = Depends(require_faculty),
    db: Session = Depends(get_db),
):
    service = BiometricService(db)
    return service.delete_reference(
        student_id_str=student_id,
        reference_id=reference_id,
        current_user=current_user,
    )


@router.get(
    "/students/{student_id}/status",
    response_model=FaceRegistrationStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Get student biometric registration status",
    description=(
        "Retrieves enrolment status, active and total reference count, enrolled pose labels, "
        "and latest model metadata. Zero raw embeddings or internal file paths exposed."
    ),
)
def get_student_biometric_status(
    student_id: str = Path(..., description="Student institutional ID e.g. 'STU2026001'"),
    current_user: User = Depends(require_faculty),
    db: Session = Depends(get_db),
):
    service = BiometricService(db)
    return service.get_registration_status(
        student_id_str=student_id,
        current_user=current_user,
    )
