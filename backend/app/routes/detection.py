"""
ReliAttend AI — Multi-Face Detection Router (Phase 7)
Exposes endpoint for multi-face detection in classroom frames.
Strictly restricted to authenticated Faculty and Admin users.
"""

from typing import Optional
from fastapi import APIRouter, Depends, File, Form, Query, UploadFile, status

from app.models.user import User
from app.routes.deps import require_faculty
from app.schemas.detection import MultiFaceDetectionResponse
from app.services.detection_service import DetectionService

router = APIRouter(prefix="/detection", tags=["Face Detection Engine"])


@router.post(
    "/detect-faces",
    response_model=MultiFaceDetectionResponse,
    status_code=status.HTTP_200_OK,
    summary="Detect faces in classroom group frame",
    description=(
        "Detects zero, one, two, or multiple human faces in a single frame. "
        "Returns bounding boxes (absolute pixels), 5-point facial landmarks, "
        "detection confidence, and per-face quality indicators. "
        "Frames with zero faces return HTTP 200 with an empty list. "
        "Does NOT perform face recognition, tracking, or filesystem storage."
    ),
)
async def detect_faces_in_frame(
    file: UploadFile = File(..., description="JPEG or PNG frame image (max 5 MB)"),
    confidence_threshold: Optional[float] = Query(
        None,
        ge=0.01,
        le=1.0,
        description="Optional YuNet score threshold (default: 0.60)",
    ),
    nms_threshold: Optional[float] = Query(
        None,
        ge=0.01,
        le=1.0,
        description="Optional Non-Maximum Suppression threshold (default: 0.30)",
    ),
    current_user: User = Depends(require_faculty),
):
    service = DetectionService()
    return await service.detect_faces_in_frame(
        file=file,
        current_user=current_user,
        confidence_threshold=confidence_threshold,
        nms_threshold=nms_threshold,
    )

