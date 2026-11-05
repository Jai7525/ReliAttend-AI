"""
ReliAttend AI — Multi-Face Tracking Engine (Phase 9 - Substage 9.2)
Orchestrates sequential frame tracking, one-to-one detection association,
Kalman motion filtering, track state transitions, and ambiguity detection.
"""

from dataclasses import dataclass, field
import logging
import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from app.ai.tracking.association import (
    associate_detections_hybrid,
    associate_detections_to_tracks,
    compute_appearance_similarity_matrix,
    compute_iou,
    compute_iou_matrix,
    reidentify_lost_tracks,
)
from app.ai.tracking.kalman import KalmanBoxTracker
from app.ai.tracking.track import FaceTrack, TrackState

logger = logging.getLogger("reliattend.tracking")


@dataclass
class DetectionInput:
    """Standardized internal representation of a detected face in a frame."""
    face_id: int  # Transient single-frame index
    bounding_box: Tuple[int, int, int, int]  # (x, y, w, h)
    confidence: float
    landmarks: Optional[Dict[str, Tuple[int, int]]] = None
    embedding: Optional[np.ndarray] = None  # 128-d float32 SFace normalized vector (in-memory only)
    is_quality_acceptable: bool = True


@dataclass
class TrackOutput:
    """Structured track output returned per frame."""
    track_id: str
    state: str  # "tentative", "confirmed", "temporarily_lost", "terminated"
    bounding_box: Tuple[int, int, int, int]
    detection_confidence: float
    total_observations: int
    consecutive_misses: int
    first_seen_frame: int
    last_seen_frame: int
    first_seen_timestamp: float
    last_seen_timestamp: float
    is_newly_created: bool = False
    is_recovered: bool = False
    is_ambiguous: bool = False
    associated_student_id: Optional[str] = None
    student_name: Optional[str] = None
    recognition_similarity: Optional[float] = None
    identity_status: str = "unassigned"  # "assigned", "provisional", "conflict", "unassigned"
    consistent_hits: int = 0
    matched_face_id: Optional[int] = None


@dataclass
class FrameTrackingResult:
    """Overall multi-face tracking result for a processed frame."""
    frame_seq: int
    timestamp: float
    active_tracks_count: int
    confirmed_tracks_count: int
    tracks: List[TrackOutput]
    unmatched_detections_count: int


class MultiFaceTracker:
    """
    In-memory multi-face tracking engine.
    Maintains temporary track identities across sequential frames using
    Kalman state extrapolation, IoU spatial association, and conservative ambiguity resolution.
    """

    def __init__(
        self,
        iou_threshold: float = 0.30,
        min_hits_to_confirm: int = 3,
        max_missing_frames: int = 15,
        max_missing_seconds: float = 3.0,
        ambiguity_iou_threshold: float = 0.20,
        appearance_threshold: float = 0.60,
        weight_spatial: float = 0.50,
        weight_appearance: float = 0.50,
        spatial_gating_iou: float = 0.05,
        ambiguity_margin: float = 0.08,
        reid_appearance_threshold: float = 0.65,
    ):
        """
        Configuration Defaults & Rationale:
        - iou_threshold = 0.30: Minimum overlap between predicted box and detection. Standard for 15-30 FPS face motion.
        - min_hits_to_confirm = 3: Requires 3 consecutive observations (~0.1-0.2s) to promote from tentative, eliminating single-frame false-positive flickers.
        - max_missing_frames = 15: Allows up to 15 missed frames (~0.5-1.0s) for temporary head turns or brief occlusions before terminating.
        - max_missing_seconds = 3.0: Absolute elapsed wall-clock limit preventing stale track reconnection over long dropouts.
        - ambiguity_iou_threshold = 0.20: If two active tracks both overlap the same detection with IoU >= 0.20, flagged as ambiguous.
        - appearance_threshold = 0.60: SFace cosine similarity threshold allowing visual recovery when IoU is low.
        - weight_spatial = 0.50, weight_appearance = 0.50: Balanced cost weighting for hybrid bipartite assignment.
        - spatial_gating_iou = 0.05: Minimum spatial plausibility floor to prevent matching across opposite corners of the screen.
        - ambiguity_margin = 0.08: Ambiguity margin between top-1 and top-2 candidate tracks to prevent forced identity swapping.
        - reid_appearance_threshold = 0.65: High-confidence similarity threshold for Stage-2 re-identification of temporarily lost tracks.
        """
        self.iou_threshold = iou_threshold
        self.min_hits_to_confirm = min_hits_to_confirm
        self.max_missing_frames = max_missing_frames
        self.max_missing_seconds = max_missing_seconds
        self.ambiguity_iou_threshold = ambiguity_iou_threshold
        self.appearance_threshold = appearance_threshold
        self.weight_spatial = weight_spatial
        self.weight_appearance = weight_appearance
        self.spatial_gating_iou = spatial_gating_iou
        self.ambiguity_margin = ambiguity_margin
        self.reid_appearance_threshold = reid_appearance_threshold

        self.tracks: Dict[str, FaceTrack] = {}
        self.next_track_number = 1
        self.last_processed_frame_seq: Optional[int] = None
        self.last_processed_timestamp: Optional[float] = None

    def _generate_track_id(self) -> str:
        """Generates a strictly unique track ID within this tracker instance."""
        tid = f"TRK-{self.next_track_number:04d}"
        self.next_track_number += 1
        return tid

    def _validate_frame_ordering(self, frame_seq: int, timestamp: float) -> None:
        """
        Guarantees strictly increasing frame sequence ordering.
        Rejects duplicate or out-of-order frame submissions.
        """
        if self.last_processed_frame_seq is not None:
            if frame_seq <= self.last_processed_frame_seq:
                raise ValueError(
                    f"Out-of-order or duplicate frame received: frame_seq={frame_seq} "
                    f"is not strictly greater than last processed frame_seq={self.last_processed_frame_seq}."
                )

        if self.last_processed_timestamp is not None:
            if timestamp < self.last_processed_timestamp:
                raise ValueError(
                    f"Negative time delta received: timestamp={timestamp} "
                    f"is earlier than last processed timestamp={self.last_processed_timestamp}."
                )

    def _detect_ambiguous_assignments(
        self,
        predicted_boxes: List[Tuple[int, int, int, int]],
        detection_boxes: List[Tuple[int, int, int, int]],
        matches: List[Tuple[int, int]],
    ) -> Set[int]:
        """
        Identifies ambiguous associations where crossing or overlapping faces
        could result in an identity swap.
        If a detection overlaps a SECOND track with IoU >= ambiguity_iou_threshold,
        that detection assignment is flagged as ambiguous.
        """
        ambiguous_match_indices: Set[int] = set()
        if not matches or len(predicted_boxes) < 2 or not detection_boxes:
            return ambiguous_match_indices

        iou_matrix = compute_iou_matrix(predicted_boxes, detection_boxes)

        for match_idx, (t_idx, d_idx) in enumerate(matches):
            # Check if any OTHER track also has significant overlap with this detection
            for other_t_idx in range(len(predicted_boxes)):
                if other_t_idx != t_idx:
                    other_iou = iou_matrix[other_t_idx, d_idx]
                    if other_iou >= self.ambiguity_iou_threshold:
                        ambiguous_match_indices.add(match_idx)
                        break

        return ambiguous_match_indices

    def update(
        self,
        detections: List[DetectionInput],
        frame_seq: int,
        timestamp: Optional[float] = None,
    ) -> FrameTrackingResult:
        """
        Processes a single sequential frame through detection-to-track association.
        Handles zero-detection frames, creates tentative tracks for new faces,
        coasts missing tracks via Kalman prediction, and purges terminated tracks.
        """
        curr_time = timestamp if timestamp is not None else time.monotonic()
        self._validate_frame_ordering(frame_seq, curr_time)

        self.last_processed_frame_seq = frame_seq
        self.last_processed_timestamp = curr_time

        # 1. Step 1: Advance Kalman motion models for all active tracks
        active_track_ids = [tid for tid, trk in self.tracks.items() if trk.is_active]
        predicted_boxes: List[Tuple[int, int, int, int]] = []

        for tid in active_track_ids:
            predicted_boxes.append(self.tracks[tid].predict())

        # 2. Step 2: Validate and sanitize detection bounding boxes and appearance features
        valid_detections: List[DetectionInput] = []
        detection_boxes: List[Tuple[int, int, int, int]] = []
        detection_embeddings: List[Optional[np.ndarray]] = []

        for det in detections:
            x, y, w, h = det.bounding_box
            if w > 0 and h > 0:
                valid_detections.append(det)
                detection_boxes.append((x, y, w, h))
                detection_embeddings.append(det.embedding)

        track_embeddings: List[List[np.ndarray]] = [
            self.tracks[tid].get_appearance_embeddings() for tid in active_track_ids
        ]

        # 3. Step 3: Solve one-to-one hybrid assignment (spatial + appearance)
        matches, unmatched_t_indices, unmatched_d_indices, ambiguous_match_indices = associate_detections_hybrid(
            track_boxes=predicted_boxes,
            detection_boxes=detection_boxes,
            track_embeddings=track_embeddings,
            detection_embeddings=detection_embeddings,
            iou_threshold=self.iou_threshold,
            appearance_threshold=self.appearance_threshold,
            weight_spatial=self.weight_spatial,
            weight_appearance=self.weight_appearance,
            spatial_gating_iou=self.spatial_gating_iou,
            ambiguity_margin=self.ambiguity_margin,
        )

        output_tracks: List[TrackOutput] = []

        # 4. Step 4: Update matched tracks
        for m_idx, (t_idx, d_idx) in enumerate(matches):
            tid = active_track_ids[t_idx]
            trk = self.tracks[tid]
            det = valid_detections[d_idx]

            was_lost = (trk.state == TrackState.TEMPORARILY_LOST)
            is_ambiguous = (m_idx in ambiguous_match_indices)

            trk.update_matched(
                bbox=det.bounding_box,
                frame_seq=frame_seq,
                timestamp=curr_time,
                detection_confidence=det.confidence,
                landmarks=det.landmarks,
                is_ambiguous=is_ambiguous,
            )

            # Safely add appearance template (protected against ambiguous or low-quality updates)
            trk.add_appearance_embedding(
                embedding=det.embedding,
                is_quality_acceptable=det.is_quality_acceptable,
                is_ambiguous=is_ambiguous,
            )

            ident = trk.identity_accumulator
            output_tracks.append(
                TrackOutput(
                    track_id=trk.track_id,
                    state=trk.state.value,
                    bounding_box=trk.current_bbox,
                    detection_confidence=trk.last_detection_confidence,
                    total_observations=trk.total_observations,
                    consecutive_misses=trk.consecutive_misses,
                    first_seen_frame=trk.first_seen_frame,
                    last_seen_frame=trk.last_seen_frame,
                    first_seen_timestamp=trk.first_seen_timestamp,
                    last_seen_timestamp=trk.last_seen_timestamp,
                    is_newly_created=False,
                    is_recovered=was_lost,
                    is_ambiguous=is_ambiguous,
                    associated_student_id=ident.assigned_student_id,
                    student_name=ident.assigned_student_name,
                    recognition_similarity=ident.recognition_similarity,
                    identity_status=ident.identity_status,
                    consistent_hits=len([o for o in ident.history if o and o.student_str_id == ident.assigned_student_id]) if ident.assigned_student_id else 0,
                    matched_face_id=det.face_id,
                )
            )

        # 5. Step 5: Stage-2 Association — Re-identification of Temporarily Lost Tracks
        # For unmatched detections with embeddings, attempt to re-identify against CONFIRMED tracks that are TEMPORARILY_LOST
        reid_matches: List[Tuple[str, int]] = []
        if unmatched_d_indices:
            lost_candidates: List[Tuple[str, List[np.ndarray], Tuple[int, int, int, int]]] = []
            for tid in active_track_ids:
                trk = self.tracks[tid]
                # Only consider tracks that were already TEMPORARILY_LOST or became missed in this frame,
                # have appearance history, and are not among the primary matches
                is_currently_matched = any(active_track_ids[t_i] == tid for t_i, _ in matches)
                if not is_currently_matched and trk.state in (TrackState.TEMPORARILY_LOST, TrackState.CONFIRMED):
                    app_embs = trk.get_appearance_embeddings()
                    if app_embs:
                        lost_candidates.append((tid, app_embs, trk.current_bbox))

            unmatched_det_candidates: List[Tuple[int, Tuple[int, int, int, int], Optional[np.ndarray]]] = [
                (d_i, valid_detections[d_i].bounding_box, valid_detections[d_i].embedding)
                for d_i in unmatched_d_indices
                if valid_detections[d_i].embedding is not None and valid_detections[d_i].is_quality_acceptable
            ]

            if lost_candidates and unmatched_det_candidates:
                reid_matches, remaining_d_indices = reidentify_lost_tracks(
                    lost_tracks=lost_candidates,
                    unmatched_detections=unmatched_det_candidates,
                    appearance_threshold=self.reid_appearance_threshold,
                    max_scale_ratio=3.5,
                    ambiguity_margin=self.ambiguity_margin,
                )
                # Filter unmatched_d_indices to exclude recovered detections
                reidentified_d_set = {d_i for _, d_i in reid_matches}
                unmatched_d_indices = [d_i for d_i in unmatched_d_indices if d_i not in reidentified_d_set]

        # Process Stage-2 Re-identified Tracks
        reidentified_track_ids: Set[str] = set()
        for r_tid, r_d_idx in reid_matches:
            r_trk = self.tracks[r_tid]
            r_det = valid_detections[r_d_idx]
            reidentified_track_ids.add(r_tid)

            # Re-initialize Kalman state at newly recovered detection location
            r_trk.kalman = KalmanBoxTracker(r_det.bounding_box)
            r_trk.current_bbox = r_det.bounding_box
            r_trk.predicted_bbox = r_det.bounding_box
            r_trk.last_seen_frame = frame_seq
            r_trk.last_seen_timestamp = curr_time
            r_trk.last_detection_confidence = r_det.confidence
            r_trk.last_landmarks = r_det.landmarks
            r_trk.total_observations += 1
            r_trk.consecutive_hits += 1
            r_trk.consecutive_misses = 0
            r_trk.state = TrackState.CONFIRMED

            # Safely append appearance template
            r_trk.add_appearance_embedding(
                embedding=r_det.embedding,
                is_quality_acceptable=r_det.is_quality_acceptable,
                is_ambiguous=False,
            )

            r_ident = r_trk.identity_accumulator
            output_tracks.append(
                TrackOutput(
                    track_id=r_trk.track_id,
                    state=r_trk.state.value,
                    bounding_box=r_trk.current_bbox,
                    detection_confidence=r_trk.last_detection_confidence,
                    total_observations=r_trk.total_observations,
                    consecutive_misses=r_trk.consecutive_misses,
                    first_seen_frame=r_trk.first_seen_frame,
                    last_seen_frame=r_trk.last_seen_frame,
                    first_seen_timestamp=r_trk.first_seen_timestamp,
                    last_seen_timestamp=r_trk.last_seen_timestamp,
                    is_newly_created=False,
                    is_recovered=True,
                    is_ambiguous=False,
                    associated_student_id=r_ident.assigned_student_id,
                    student_name=r_ident.assigned_student_name,
                    recognition_similarity=r_ident.recognition_similarity,
                    identity_status=r_ident.identity_status,
                    consistent_hits=len([o for o in r_ident.history if o and o.student_str_id == r_ident.assigned_student_id]) if r_ident.assigned_student_id else 0,
                    matched_face_id=r_det.face_id,
                )
            )

        # 6. Step 6: Update unmatched tracks (coasting / missing)
        for t_idx in unmatched_t_indices:
            tid = active_track_ids[t_idx]
            if tid in reidentified_track_ids:
                # Track was successfully recovered in Stage 2; do not mark missed
                continue

            trk = self.tracks[tid]
            trk.update_missed(frame_seq=frame_seq, timestamp=curr_time)

            u_ident = trk.identity_accumulator
            output_tracks.append(
                TrackOutput(
                    track_id=trk.track_id,
                    state=trk.state.value,
                    bounding_box=trk.current_bbox,
                    detection_confidence=trk.last_detection_confidence,
                    total_observations=trk.total_observations,
                    consecutive_misses=trk.consecutive_misses,
                    first_seen_frame=trk.first_seen_frame,
                    last_seen_frame=trk.last_seen_frame,
                    first_seen_timestamp=trk.first_seen_timestamp,
                    last_seen_timestamp=trk.last_seen_timestamp,
                    is_newly_created=False,
                    is_recovered=False,
                    is_ambiguous=False,
                    associated_student_id=u_ident.assigned_student_id,
                    student_name=u_ident.assigned_student_name,
                    recognition_similarity=u_ident.recognition_similarity,
                    identity_status=u_ident.identity_status,
                    consistent_hits=len([o for o in u_ident.history if o and o.student_str_id == u_ident.assigned_student_id]) if u_ident.assigned_student_id else 0,
                )
            )

        # 7. Step 7: Create tentative tracks for unmatched detections
        for d_idx in unmatched_d_indices:
            det = valid_detections[d_idx]
            new_tid = self._generate_track_id()

            new_track = FaceTrack(
                track_id=new_tid,
                initial_bbox=det.bounding_box,
                frame_seq=frame_seq,
                timestamp=curr_time,
                detection_confidence=det.confidence,
                landmarks=det.landmarks,
                min_hits_to_confirm=self.min_hits_to_confirm,
                max_missing_frames=self.max_missing_frames,
                max_missing_seconds=self.max_missing_seconds,
            )
            # Add initial appearance embedding if quality is acceptable
            new_track.add_appearance_embedding(
                embedding=det.embedding,
                is_quality_acceptable=det.is_quality_acceptable,
                is_ambiguous=False,
            )
            self.tracks[new_tid] = new_track

            output_tracks.append(
                TrackOutput(
                    track_id=new_track.track_id,
                    state=new_track.state.value,
                    bounding_box=new_track.current_bbox,
                    detection_confidence=new_track.last_detection_confidence,
                    total_observations=new_track.total_observations,
                    consecutive_misses=new_track.consecutive_misses,
                    first_seen_frame=new_track.first_seen_frame,
                    last_seen_frame=new_track.last_seen_frame,
                    first_seen_timestamp=new_track.first_seen_timestamp,
                    last_seen_timestamp=new_track.last_seen_timestamp,
                    is_newly_created=True,
                    is_recovered=False,
                    is_ambiguous=False,
                    associated_student_id=None,
                    student_name=None,
                    recognition_similarity=None,
                    identity_status="unassigned",
                    consistent_hits=0,
                    matched_face_id=det.face_id,
                )
            )

        # 8. Step 8: Memory cleanup: purge terminated tracks from active dictionary
        terminated_ids = [tid for tid, trk in self.tracks.items() if trk.state == TrackState.TERMINATED]
        for tid in terminated_ids:
            del self.tracks[tid]

        # Sort outputs by track_id for predictable debugging & testing
        output_tracks.sort(key=lambda t: t.track_id)

        active_count = sum(1 for t in output_tracks if t.state != TrackState.TERMINATED.value)
        confirmed_count = sum(1 for t in output_tracks if t.state == TrackState.CONFIRMED.value)

        return FrameTrackingResult(
            frame_seq=frame_seq,
            timestamp=curr_time,
            active_tracks_count=active_count,
            confirmed_tracks_count=confirmed_count,
            tracks=output_tracks,
            unmatched_detections_count=len(unmatched_d_indices),
        )

    def get_active_tracks(self) -> List[TrackOutput]:
        """Returns all currently active (non-terminated) tracks in the tracker."""
        active_list: List[TrackOutput] = []
        for tid, trk in self.tracks.items():
            if trk.state != TrackState.TERMINATED:
                active_list.append(
                    TrackOutput(
                        track_id=trk.track_id,
                        state=trk.state.value,
                        bounding_box=trk.current_bbox,
                        detection_confidence=trk.last_detection_confidence,
                        total_observations=trk.total_observations,
                        consecutive_misses=trk.consecutive_misses,
                        first_seen_frame=trk.first_seen_frame,
                        last_seen_frame=trk.last_seen_frame,
                        first_seen_timestamp=trk.first_seen_timestamp,
                        last_seen_timestamp=trk.last_seen_timestamp,
                        is_newly_created=False,
                        is_recovered=False,
                        is_ambiguous=False,
                        associated_student_id=trk.identity_accumulator.assigned_student_id,
                        student_name=trk.identity_accumulator.assigned_student_name,
                        recognition_similarity=trk.identity_accumulator.recognition_similarity,
                        identity_status=trk.identity_accumulator.identity_status,
                        consistent_hits=len([o for o in trk.identity_accumulator.history if o and o.student_str_id == trk.identity_accumulator.assigned_student_id]) if trk.identity_accumulator.assigned_student_id else 0,
                    )
                )
        active_list.sort(key=lambda t: t.track_id)
        return active_list

    def terminate_all(self) -> None:
        """Transitions all tracks to TERMINATED and empties track memory."""
        for trk in self.tracks.values():
            trk.terminate()
        self.tracks.clear()


