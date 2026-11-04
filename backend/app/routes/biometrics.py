"""
ReliAttend AI — Biometric & Face Registration Routes (Phase 6)
Exposes endpoints for reference face registration and status inquiries.
Guards endpoints with authentication and scoped RBAC.
"""

from fastapi import APIRouter, Depends, File, Path, UploadFile, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.user import User
from app.routes.deps import require_faculty
from app.schemas.biometrics import (
    FaceRegistrationResponse,
    FaceRegistrationStatusResponse,
)
from app.services.biometric_service import BiometricService

router = APIRouter(prefix="/biometrics", tags=["Biometric Management"])


@router.post(
    "/students/{student_id}/register-face",
    response_model=FaceRegistrationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register student reference face image",
    description=(
        "Registers a student's reference face image using multipart/form-data. "
        "Enforces active consent, single-face detection, blur/brightness checks, "
        "128-d SFace feature extraction, and single-active-reference policy. "
        "Admins have institutional access; Faculty can register students assigned to their class."
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


@router.get(
    "/students/{student_id}/status",
    response_model=FaceRegistrationStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Get student biometric registration status",
    description=(
        "Retrieves enrolment status, active and total reference count, and latest model metadata. "
        "Never exposes raw embedding vectors or internal image paths."
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

