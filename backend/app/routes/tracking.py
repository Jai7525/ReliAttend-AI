"""
ReliAttend AI — Multi-Face Tracking Router (Phase 9 - Substage 9.3)
Exposes endpoints for creating tracking sessions, streaming/ingesting sequential
frames, querying active track states, and terminating sessions.
Strictly restricted to Admin and assigned Faculty.
"""

from typing import Optional
from fastapi import APIRouter, Depends, File, Form, Path, Query, UploadFile, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.user import User
from app.routes.deps import require_faculty
from app.schemas.tracking import (
    CreateTrackingSessionRequest,
    EndTrackingSessionResponse,
    ProcessFrameTrackingResponse,
    SessionTracksListResponse,
    TrackingSessionResponse,
)
from app.services.tracking_service import TrackingService

router = APIRouter(prefix="/tracking", tags=["Multi-Face Tracking Engine"])


@router.post(
    "/sessions",
    response_model=TrackingSessionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create in-memory multi-face tracking session",
    description=(
        "Initializes a new multi-face tracking session for a specific class section. "
        "Restricted to Admin or Faculty assigned to the class section. "
        "Returns session ID and tracker configuration parameters."
    ),
)
def create_tracking_session(
    request: CreateTrackingSessionRequest,
    current_user: User = Depends(require_faculty),
    db: Session = Depends(get_db),
):
    service = TrackingService(db=db)
    return service.create_session(request=request, current_user=current_user)


@router.post(
    "/sessions/{session_id}/process-frame",
    response_model=ProcessFrameTrackingResponse,
    status_code=status.HTTP_200_OK,
    summary="Process sequential image frame in tracking session",
    description=(
        "Ingests an uploaded classroom frame image, runs YuNet face detection, "
        "and updates the stateful multi-face tracker. "
        "Requires frame_seq strictly greater than previous processed frame. "
        "Zero images or biometric embeddings are stored or exposed."
    ),
)
async def process_tracking_frame(
    session_id: str = Path(..., description="Unique tracking session identifier e.g. 'trk_sess_abc123'"),
    frame_seq: int = Form(..., description="Monotonically increasing 0-indexed frame sequence number"),
    timestamp: Optional[float] = Form(None, description="Optional frame capture Unix timestamp"),
    file: UploadFile = File(..., description="JPEG or PNG frame image (max 5 MB)"),
    current_user: User = Depends(require_faculty),
    db: Session = Depends(get_db),
):
    service = TrackingService(db=db)
    return await service.process_frame(
        session_id=session_id,
        frame_seq=frame_seq,
        timestamp=timestamp,
        file=file,
        current_user=current_user,
    )


@router.get(
    "/sessions/{session_id}/tracks",
    response_model=SessionTracksListResponse,
    status_code=status.HTTP_200_OK,
    summary="List active face tracks in session",
    description=(
        "Retrieves all currently active (tentative, confirmed, temporarily lost) "
        "tracks in the specified session, their bounding boxes, and observation statistics."
    ),
)
def get_session_tracks(
    session_id: str = Path(..., description="Tracking session identifier"),
    current_user: User = Depends(require_faculty),
    db: Session = Depends(get_db),
):
    service = TrackingService(db=db)
    return service.get_tracks(session_id=session_id, current_user=current_user)


@router.delete(
    "/sessions/{session_id}",
    response_model=EndTrackingSessionResponse,
    status_code=status.HTTP_200_OK,
    summary="Terminate tracking session",
    description="Explicitly ends the tracking session and frees in-memory tracker resources.",
)
def end_tracking_session(
    session_id: str = Path(..., description="Tracking session identifier to terminate"),
    current_user: User = Depends(require_faculty),
    db: Session = Depends(get_db),
):
    service = TrackingService(db=db)
    return service.end_session(session_id=session_id, current_user=current_user)

