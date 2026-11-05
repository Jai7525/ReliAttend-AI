"""
ReliAttend AI — Multi-Face Tracking Schemas (Phase 9 - Substage 9.3)
Defines request and response schemas for tracking sessions and frame-by-frame
multi-face tracking updates.
Privacy by design: maintains track trajectories without exposing raw biometric embeddings.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field

from app.schemas.detection import BoundingBoxSchema


class CreateTrackingSessionRequest(BaseModel):
    """Payload to initiate a stateful multi-face tracking session."""
    class_section_id: int = Field(..., description="Target class section ID e.g. 1")
    iou_threshold: Optional[float] = Field(
        default=0.30,
        ge=0.05,
        le=0.90,
        description="Minimum IoU overlap threshold for spatial association (default: 0.30)",
    )
    min_hits_to_confirm: Optional[int] = Field(
        default=3,
        ge=1,
        le=20,
        description="Consecutive observations required to confirm track (default: 3)",
    )
    max_missing_frames: Optional[int] = Field(
        default=15,
        ge=1,
        le=100,
        description="Maximum consecutive missed frames before track termination (default: 15)",
    )
    max_missing_seconds: Optional[float] = Field(
        default=3.0,
        ge=0.5,
        le=30.0,
        description="Maximum wall-clock seconds before track termination (default: 3.0)",
    )
    ambiguity_iou_threshold: Optional[float] = Field(
        default=0.20,
        ge=0.05,
        le=0.80,
        description="IoU overlap threshold indicating ambiguous crossing paths (default: 0.20)",
    )
    appearance_threshold: Optional[float] = Field(
        default=0.60,
        ge=0.10,
        le=0.95,
        description="Minimum SFace cosine similarity allowing appearance-assisted association (default: 0.60)",
    )
    weight_spatial: Optional[float] = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
        description="Weight given to spatial IoU cost in hybrid association (default: 0.50)",
    )
    weight_appearance: Optional[float] = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
        description="Weight given to appearance cosine cost in hybrid association (default: 0.50)",
    )


class TrackingConfigSchema(BaseModel):
    """Echoed configuration parameters active for this tracking session."""
    iou_threshold: float
    min_hits_to_confirm: int
    max_missing_frames: int
    max_missing_seconds: float
    ambiguity_iou_threshold: float
    appearance_threshold: float = 0.60
    weight_spatial: float = 0.50
    weight_appearance: float = 0.50


class TrackingSessionResponse(BaseModel):
    """Metadata response returned upon session creation."""
    session_id: str = Field(..., description="Unique in-memory tracking session identifier")
    class_section_id: int = Field(..., description="Associated class section ID")
    created_at: float = Field(..., description="Session creation Unix timestamp in seconds")
    config: TrackingConfigSchema = Field(..., description="Active tracker configuration")

    model_config = ConfigDict(from_attributes=True)


class RecognizedStudentMetadata(BaseModel):
    """Privacy-conscious student recognition metadata associated with a track."""
    student_id_masked: str = Field(..., description="Masked institutional student ID e.g. 'STU...001'")
    student_name: Optional[str] = Field(None, description="Full student name (available to authorized faculty/admin)")
    recognition_similarity: float = Field(..., description="Aggregated cosine similarity score against enrolled reference [0.0, 1.0]")
    identity_status: str = Field(
        ...,
        description="Recognition status: 'assigned' (confirmed), 'provisional' (accumulating), 'conflict', or 'unassigned'",
    )
    consistent_hits: int = Field(..., description="Number of consistent recognition matches in the evidence window")


class TrackItemResponse(BaseModel):
    """Per-track spatial and lifecycle state for a frame or query."""
    track_id: str = Field(..., description="Persistent track identifier across frames e.g. 'TRK-0001'")
    state: str = Field(
        ...,
        description="Current track lifecycle state ('tentative', 'confirmed', 'temporarily_lost', 'terminated')",
    )
    bounding_box: BoundingBoxSchema = Field(..., description="Smoothed or predicted bounding box (x, y, w, h)")
    detection_confidence: float = Field(..., description="Detector confidence score from latest hit")
    total_observations: int = Field(..., description="Total successful frame associations for this track")
    consecutive_misses: int = Field(..., description="Number of consecutive frames without a direct detection match")
    first_seen_frame: int = Field(..., description="Frame sequence index when this track was initiated")
    last_seen_frame: int = Field(..., description="Frame sequence index when this track was last updated/predicted")
    first_seen_timestamp: float = Field(..., description="Unix timestamp of first observation")
    last_seen_timestamp: float = Field(..., description="Unix timestamp of latest observation")
    is_newly_created: bool = Field(default=False, description="True if track was initiated in this specific frame")
    is_recovered: bool = Field(default=False, description="True if track re-associated after being temporarily lost")
    is_ambiguous: bool = Field(default=False, description="True if track is spatially ambiguous with another track")
    associated_student_id: Optional[str] = Field(
        default=None,
        description="Masked or full student identifier if assigned, else None",
    )
    recognized_student: Optional[RecognizedStudentMetadata] = Field(
        default=None,
        description="Detailed recognition resolution metadata if available",
    )


class ProcessFrameTrackingResponse(BaseModel):
    """Response returned after ingesting and tracking a single image frame."""
    session_id: str = Field(..., description="Active tracking session identifier")
    frame_seq: int = Field(..., description="Processed sequence index of this frame")
    timestamp: float = Field(..., description="Client or server timestamp for this frame")
    active_tracks_count: int = Field(..., description="Number of tracks currently active (tentative, confirmed, lost)")
    confirmed_tracks_count: int = Field(..., description="Number of confirmed tracks in this frame")
    unmatched_detections_count: int = Field(..., description="Number of detections not associated with existing tracks")
    tracks: List[TrackItemResponse] = Field(
        default_factory=list,
        description="Track states updated for this frame (excludes terminated tracks)",
    )


class SessionTracksListResponse(BaseModel):
    """Response returned when querying current active tracks in a session."""
    session_id: str = Field(..., description="Tracking session identifier")
    active_tracks_count: int = Field(..., description="Count of currently active tracks")
    tracks: List[TrackItemResponse] = Field(default_factory=list, description="Active tracks list")


class EndTrackingSessionResponse(BaseModel):
    """Response returned upon terminating a tracking session."""
    session_id: str = Field(..., description="Terminated session identifier")
    message: str = Field(..., description="Confirmation message")
    total_tracks_created: int = Field(..., description="Total distinct tracks created throughout the session")
    duration_seconds: float = Field(..., description="Total elapsed session lifespan in seconds")

