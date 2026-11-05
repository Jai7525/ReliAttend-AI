"""
ReliAttend AI — Tracking Module Package (Phase 9)
Provides Kalman motion filtering, spatial association, track lifecycle, and multi-face tracking.
"""

from app.ai.tracking.association import (
    associate_detections_to_tracks,
    compute_iou,
    compute_iou_matrix,
    linear_assignment,
)
from app.ai.tracking.kalman import KalmanBoxTracker
from app.ai.tracking.track import FaceTrack, TrackObservation, TrackState
from app.ai.tracking.tracker import (
    DetectionInput,
    FrameTrackingResult,
    MultiFaceTracker,
    TrackOutput,
)

__all__ = [
    "KalmanBoxTracker",
    "compute_iou",
    "compute_iou_matrix",
    "linear_assignment",
    "associate_detections_to_tracks",
    "FaceTrack",
    "TrackState",
    "TrackObservation",
    "DetectionInput",
    "TrackOutput",
    "FrameTrackingResult",
    "MultiFaceTracker",
]


