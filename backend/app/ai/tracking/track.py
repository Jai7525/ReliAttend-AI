"""
ReliAttend AI — Individual Face Track Lifecycle & State (Phase 9 - Substage 9.2)
Maintains state machine, bounding-box motion estimation, temporal counters,
and clean separation between tracking continuity and recognition identity.
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Deque, Dict, List, Optional, Tuple
import numpy as np

from app.ai.tracking.kalman import KalmanBoxTracker


class TrackState(str, Enum):
    """
    Four-stage tracking lifecycle states:
    - TENTATIVE: Initial candidate track awaiting temporal confirmation.
    - CONFIRMED: Established face track consistently observed across consecutive frames.
    - TEMPORARILY_LOST: Missed detection in current frame; position coasted by Kalman prediction.
    - TERMINATED: Inactive track that exceeded missing frame count or elapsed time limit.
    """
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    TEMPORARILY_LOST = "temporarily_lost"
    TERMINATED = "terminated"


@dataclass
class TrackObservation:
    """Snapshot of a single detection update on a track."""
    frame_seq: int
    timestamp: float  # Monotonic seconds or epoch timestamp
    bbox: Tuple[int, int, int, int]
    detection_confidence: float
    landmarks: Optional[Dict[str, Tuple[int, int]]] = None


@dataclass
class RecognitionObservation:
    """Snapshot of a single eligible recognition match."""
    student_str_id: str
    student_name: str
    similarity: float
    frame_seq: int
    timestamp: float


class TrackIdentityAccumulator:
    """
    Maintains rolling recognition evidence for an individual FaceTrack.
    Enforces the rule: require at least 3 consistent matched observations
    within a rolling window of 5 eligible frames to assign student identity.
    """

    def __init__(self, min_consistent_hits: int = 3, window_size: int = 5):
        self.min_consistent_hits = min_consistent_hits
        self.window_size = window_size
        self.history: Deque[Optional[RecognitionObservation]] = deque(maxlen=window_size)
        self.assigned_student_id: Optional[str] = None
        self.assigned_student_name: Optional[str] = None
        self.recognition_similarity: Optional[float] = None
        self.identity_status: str = "unassigned"  # "unassigned", "provisional", "assigned", "conflict"

    def add_observation(
        self,
        observation: Optional[RecognitionObservation],
        track_state: TrackState,
        is_ambiguous: bool,
    ) -> None:
        """
        Processes recognition evidence for the current frame:
        - If frame is ambiguous (crossing paths), pause accumulation (no-op).
        - If track is TENTATIVE, do not assign identity (can accumulate provisional or remain unassigned).
        """
        if is_ambiguous:
            # Ambiguous tracking frames pause recognition accumulation without adding evidence
            return

        self.history.append(observation)
        self._evaluate_identity(track_state)

    def _evaluate_identity(self, track_state: TrackState) -> None:
        """Evaluates whether consistent hits criteria are met."""
        valid_obs = [obs for obs in self.history if obs is not None]

        if not valid_obs:
            if self.identity_status != "assigned":
                self.identity_status = "unassigned"
                self.recognition_similarity = None
            return

        # Count frequencies of student IDs in the rolling window
        counts: Dict[str, List[RecognitionObservation]] = {}
        for obs in valid_obs:
            counts.setdefault(obs.student_str_id, []).append(obs)

        # Find top candidate
        top_student, top_list = max(counts.items(), key=lambda item: len(item[1]))

        # Calculate average similarity for top candidate
        avg_sim = float(np.mean([o.similarity for o in top_list]))
        self.recognition_similarity = round(avg_sim, 4)

        # Check if criteria met (at least min_consistent_hits)
        if len(top_list) >= self.min_consistent_hits:
            if track_state == TrackState.CONFIRMED or self.identity_status == "assigned":
                self.assigned_student_id = top_student
                self.assigned_student_name = top_list[-1].student_name
                self.identity_status = "assigned"
            else:
                # Track is tentative: cannot receive assigned identity yet
                self.identity_status = "provisional"
        else:
            if self.identity_status != "assigned":
                self.identity_status = "provisional"

    def mark_conflict(self) -> None:
        """Flags identity conflict (e.g. another active track holds this student identity)."""
        self.identity_status = "conflict"

    def clear(self) -> None:
        """Purges all accumulated recognition state."""
        self.history.clear()
        self.assigned_student_id = None
        self.assigned_student_name = None
        self.recognition_similarity = None
        self.identity_status = "unassigned"


class FaceTrack:
    """
    State manager for a single tracked human face across consecutive frames.
    Maintains Kalman motion filter, hit/miss streaks, and temporal lifecycle.
    """

    def __init__(
        self,
        track_id: str,
        initial_bbox: Tuple[int, int, int, int],
        frame_seq: int,
        timestamp: float,
        detection_confidence: float,
        landmarks: Optional[Dict[str, Tuple[int, int]]] = None,
        min_hits_to_confirm: int = 3,
        max_missing_frames: int = 15,
        max_missing_seconds: float = 3.0,
    ):
        self.track_id = track_id
        self.state = TrackState.TENTATIVE
        self.min_hits_to_confirm = min_hits_to_confirm
        self.max_missing_frames = max_missing_frames
        self.max_missing_seconds = max_missing_seconds

        # Kalman Filter for 2D bounding-box motion prediction
        self.kalman = KalmanBoxTracker(initial_bbox)
        self.current_bbox = initial_bbox
        self.predicted_bbox = initial_bbox

        # Temporal and frame sequence counters
        self.first_seen_frame = frame_seq
        self.last_seen_frame = frame_seq
        self.first_seen_timestamp = timestamp
        self.last_seen_timestamp = timestamp

        self.total_observations = 1
        self.consecutive_hits = 1
        self.consecutive_misses = 0

        # Latest detection attributes
        self.last_detection_confidence = detection_confidence
        self.last_landmarks = landmarks

        # Bounded Appearance Feature History (Substage 9.5)
        # Stores at most 3 normalized 128-d SFace float32 vectors in-memory only.
        # Zero disk/database persistence.
        self.appearance_history: Deque[np.ndarray] = deque(maxlen=3)

        # Integrated Recognition State Manager (Substage 9.7)
        self.identity_accumulator = TrackIdentityAccumulator(min_consistent_hits=3, window_size=5)

    def add_appearance_embedding(
        self,
        embedding: Optional[np.ndarray],
        is_quality_acceptable: bool = True,
        is_ambiguous: bool = False,
    ) -> bool:
        """
        Safely appends a new 128-d appearance template to the track's bounded ring buffer.
        
        Strict Quality & Integrity Safeguards:
        1. Reject if embedding is None or invalid length (must be 128 float32).
        2. Reject if face detection quality was flagged unacceptable (low resolution/blurred).
        3. Reject if detection was flagged ambiguous (e.g. crossing faces) to prevent template contamination.
        
        Returns:
            True if embedding was accepted into memory; False otherwise.
        """
        if embedding is None or embedding.shape != (128,):
            return False

        if not is_quality_acceptable:
            return False

        if is_ambiguous:
            return False

        self.appearance_history.append(embedding.astype(np.float32))
        return True

    def get_appearance_embeddings(self) -> List[np.ndarray]:
        """Returns the list of stored appearance templates (at most 3)."""
        return list(self.appearance_history)

    def predict(self) -> Tuple[int, int, int, int]:
        """
        Extrapolates track position by one frame using the Kalman filter.
        Returns predicted bounding box (x, y, w, h).
        """
        self.predicted_bbox = self.kalman.predict()
        self.current_bbox = self.predicted_bbox
        return self.predicted_bbox

    def update_matched(
        self,
        bbox: Tuple[int, int, int, int],
        frame_seq: int,
        timestamp: float,
        detection_confidence: float,
        landmarks: Optional[Dict[str, Tuple[int, int]]] = None,
        is_ambiguous: bool = False,
    ) -> None:
        """
        Updates the track state when an associated detection is successfully matched.
        If is_ambiguous is True, Kalman measurement update is bypassed (coasting on prediction)
        to prevent corrupted bounding-box velocities and identity swaps during crossing paths.
        """
        if not is_ambiguous:
            self.kalman.update(bbox)
            self.current_bbox = bbox
        else:
            # During ambiguous crossing paths, preserve Kalman state without innovation
            # The current_bbox retains the predicted bbox to prevent jumping/swapping
            self.current_bbox = self.predicted_bbox

        self.last_seen_frame = frame_seq
        self.last_seen_timestamp = timestamp
        self.last_detection_confidence = detection_confidence
        self.last_landmarks = landmarks

        self.total_observations += 1
        self.consecutive_hits += 1
        self.consecutive_misses = 0

        # Lifecycle State Transitions:
        # 1. TENTATIVE -> CONFIRMED (when consecutive_hits >= min_hits_to_confirm)
        if self.state == TrackState.TENTATIVE and self.consecutive_hits >= self.min_hits_to_confirm:
            self.state = TrackState.CONFIRMED

        # 2. TEMPORARILY_LOST -> CONFIRMED (recovery)
        elif self.state == TrackState.TEMPORARILY_LOST:
            self.state = TrackState.CONFIRMED

    def update_missed(self, frame_seq: int, timestamp: float) -> None:
        """
        Updates the track state when no associated detection was found in current frame.
        """
        self.consecutive_misses += 1
        self.consecutive_hits = 0

        elapsed_since_seen = timestamp - self.last_seen_timestamp

        # Lifecycle State Transitions:
        # 1. TENTATIVE track missing early -> terminate immediately to reject false-positive noise
        if self.state == TrackState.TENTATIVE:
            self.state = TrackState.TERMINATED

        # 2. CONFIRMED track missed -> become TEMPORARILY_LOST
        elif self.state == TrackState.CONFIRMED:
            self.state = TrackState.TEMPORARILY_LOST

        # 3. Exceeded maximum allowable missed frames or elapsed time limit -> TERMINATE
        if self.state == TrackState.TEMPORARILY_LOST:
            if (
                self.consecutive_misses > self.max_missing_frames
                or elapsed_since_seen > self.max_missing_seconds
            ):
                self.state = TrackState.TERMINATED

    def mark_terminated(self) -> None:
        """Explicit termination of the track."""
        self.state = TrackState.TERMINATED

    @property
    def is_active(self) -> bool:
        """Returns True if the track is in an active tracking state."""
        return self.state in (TrackState.TENTATIVE, TrackState.CONFIRMED, TrackState.TEMPORARILY_LOST)

