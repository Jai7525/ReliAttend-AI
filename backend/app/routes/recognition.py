"""
ReliAttend AI — Multi-Face Recognition Router (Phase 8)
Exposes endpoint for 1:N face matching in classroom attendance frames.
Strictly restricted to Admin and assigned Faculty.
"""

from typing import Optional
from fastapi import APIRouter, Depends, File, Path, Query, UploadFile, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.user import User
from app.routes.deps import require_faculty
from app.schemas.recognition import FrameRecognitionResponse
from app.services.recognition_service import RecognitionService

router = APIRouter(prefix="/recognition", tags=["Face Recognition Engine"])


@router.post(
    "/sections/{class_section_id}/recognize-frame",
    response_model=FrameRecognitionResponse,
    status_code=status.HTTP_200_OK,
    summary="Recognize faces in classroom group frame against section gallery",
    description=(
        "Detects and extracts 128-d embeddings for all faces in a group frame, "
        "matching each face against the gallery of enrolled students in the section. "
        "Applies experimental similarity threshold (0.65) and ambiguity margin (0.05). "
        "Flags duplicate student assignments and ambiguous collision matches as uncertain. "
        "Returns HTTP 200 with empty list when no faces are detected."
    ),
)
async def recognize_frame_for_section(
    class_section_id: int = Path(..., description="Target class section ID e.g. 1"),
    file: UploadFile = File(..., description="JPEG or PNG frame image (max 5 MB)"),
    similarity_threshold: Optional[float] = Query(
        None,
        ge=0.01,
        le=1.0,
        description="Optional experimental cosine similarity threshold (default: 0.65)",
    ),
    ambiguity_margin: Optional[float] = Query(
        None,
        ge=0.0,
        le=0.5,
        description="Optional ambiguity margin between top-1 and top-2 candidates (default: 0.05)",
    ),
    current_user: User = Depends(require_faculty),
    db: Session = Depends(get_db),
):
    service = RecognitionService(db=db)
    return await service.recognize_frame_for_section(
        class_section_id=class_section_id,
        file=file,
        current_user=current_user,
        similarity_threshold=similarity_threshold,
        ambiguity_margin=ambiguity_margin,
    )

