"""
ReliAttend AI — Multi-Face Tracker Engine Unit Tests (Phase 9 - Substage 9.2)
Comprehensive deterministic synthetic tests verifying:
- Single face and multiple faces continuous tracking.
- Tentative -> Confirmed -> Temporarily Lost -> Terminated transitions.
- Temporary occlusion and recovery.
- Expiration by missing-frame limit and elapsed-time limit.
- Crossing/overlapping face ambiguity detection.
- Monotonic frame sequence and duplicate/out-of-order rejection.
- Independent tracker instances and unique Track ID guarantees.
- Empty frame and invalid bounding-box handling.
"""

import time
import pytest

from app.ai.tracking import (
    DetectionInput,
    MultiFaceTracker,
    TrackState,
)


def _det(face_id: int, x: int, y: int, w: int = 50, h: int = 60, conf: float = 0.95) -> DetectionInput:
    """Helper to generate standardized synthetic DetectionInput."""
    return DetectionInput(face_id=face_id, bounding_box=(x, y, w, h), confidence=conf)


# ==============================================================================
# 1. CONTINUOUS TRACKING & PROMOTION (TENTATIVE -> CONFIRMED)
# ==============================================================================

def test_single_face_tracking_and_confirmation():
    """Face enters as tentative on Frame 1, becomes confirmed after min_hits (3 frames)."""
    tracker = MultiFaceTracker(min_hits_to_confirm=3)

    # Frame 1: Initial detection creates tentative track
    res1 = tracker.update([_det(1, 100, 100)], frame_seq=1, timestamp=1.0)
    assert res1.active_tracks_count == 1
    assert res1.confirmed_tracks_count == 0
    t1 = res1.tracks[0]
    assert t1.track_id == "TRK-0001"
    assert t1.state == TrackState.TENTATIVE.value
    assert t1.is_newly_created is True
    assert t1.total_observations == 1

    # Frame 2: Continuing observation
    res2 = tracker.update([_det(1, 102, 101)], frame_seq=2, timestamp=1.1)
    assert res2.tracks[0].track_id == "TRK-0001"
    assert res2.tracks[0].state == TrackState.TENTATIVE.value
    assert res2.tracks[0].total_observations == 2

    # Frame 3: 3rd observation promotes to CONFIRMED
    res3 = tracker.update([_det(1, 104, 102)], frame_seq=3, timestamp=1.2)
    assert res3.confirmed_tracks_count == 1
    assert res3.tracks[0].track_id == "TRK-0001"
    assert res3.tracks[0].state == TrackState.CONFIRMED.value
    assert res3.tracks[0].total_observations == 3


def test_multiple_faces_moving_independently():
    """Two faces move independently without Track ID collisions or swapping."""
    tracker = MultiFaceTracker(min_hits_to_confirm=2)

    # Frame 1: Face A at (100, 100), Face B at (400, 400)
    res1 = tracker.update([_det(1, 100, 100), _det(2, 400, 400)], frame_seq=1, timestamp=1.0)
    assert res1.active_tracks_count == 2
    assert res1.tracks[0].track_id == "TRK-0001"
    assert res1.tracks[1].track_id == "TRK-0002"

    # Frame 2: Both move slightly
    res2 = tracker.update([_det(1, 105, 102), _det(2, 405, 398)], frame_seq=2, timestamp=1.1)
    assert res2.confirmed_tracks_count == 2
    # Verify exact one-to-one track association
    tA = next(t for t in res2.tracks if t.track_id == "TRK-0001")
    tB = next(t for t in res2.tracks if t.track_id == "TRK-0002")
    assert abs(tA.bounding_box[0] - 105) <= 2
    assert abs(tB.bounding_box[0] - 405) <= 2


# ==============================================================================
# 2. OCCLUSION, TEMPORARILY LOST, AND RECOVERY
# ==============================================================================

def test_temporary_missed_detection_and_recovery():
    """Confirmed track misses 1 frame (temporarily lost) and recovers on next frame."""
    tracker = MultiFaceTracker(min_hits_to_confirm=2, max_missing_frames=5)

    # Confirm track over frames 1 & 2
    tracker.update([_det(1, 100, 100)], frame_seq=1, timestamp=1.0)
    tracker.update([_det(1, 102, 100)], frame_seq=2, timestamp=1.1)

    # Frame 3: Face temporarily occluded / not detected (empty detections)
    res3 = tracker.update([], frame_seq=3, timestamp=1.2)
    assert res3.active_tracks_count == 1
    assert res3.confirmed_tracks_count == 0  # Is temporarily lost, not confirmed
    lost_t = res3.tracks[0]
    assert lost_t.track_id == "TRK-0001"
    assert lost_t.state == TrackState.TEMPORARILY_LOST.value
    assert lost_t.consecutive_misses == 1

    # Frame 4: Face reappears near predicted location -> RECOVERED
    res4 = tracker.update([_det(1, 105, 100)], frame_seq=4, timestamp=1.3)
    recov_t = res4.tracks[0]
    assert recov_t.track_id == "TRK-0001"
    assert recov_t.state == TrackState.CONFIRMED.value
    assert recov_t.is_recovered is True
    assert recov_t.consecutive_misses == 0


# ==============================================================================
# 3. TRACK EXPIRATION & TERMINATION LIMITS
# ==============================================================================

def test_track_expiration_by_missing_frame_limit():
    """Track terminates and is purged after exceeding max_missing_frames."""
    max_missing = 4
    tracker = MultiFaceTracker(min_hits_to_confirm=2, max_missing_frames=max_missing)

    tracker.update([_det(1, 100, 100)], frame_seq=1, timestamp=1.0)
    tracker.update([_det(1, 100, 100)], frame_seq=2, timestamp=1.1)

    # Miss 1, 2, 3, 4 frames -> temporarily lost
    for seq in range(3, 7):
        res = tracker.update([], frame_seq=seq, timestamp=1.1 + (seq - 2) * 0.1)
        assert res.active_tracks_count == 1
        assert res.tracks[0].state == TrackState.TEMPORARILY_LOST.value

    # Miss 5th frame (> max_missing_frames of 4) -> TERMINATED
    res_term = tracker.update([], frame_seq=7, timestamp=1.6)
    assert res_term.tracks[0].state == TrackState.TERMINATED.value

    # On next frame 8, the terminated track is fully purged from memory
    res_next = tracker.update([], frame_seq=8, timestamp=1.7)
    assert res_next.active_tracks_count == 0
    assert len(tracker.tracks) == 0


def test_track_expiration_by_elapsed_time_limit():
    """Track terminates when elapsed wall-clock seconds exceed max_missing_seconds."""
    tracker = MultiFaceTracker(min_hits_to_confirm=2, max_missing_frames=100, max_missing_seconds=2.0)

    tracker.update([_det(1, 100, 100)], frame_seq=1, timestamp=10.0)
    tracker.update([_det(1, 100, 100)], frame_seq=2, timestamp=10.1)

    # Miss detection with elapsed time = 2.5 seconds (> 2.0s limit)
    res = tracker.update([], frame_seq=3, timestamp=12.6)
    assert res.tracks[0].state == TrackState.TERMINATED.value


def test_tentative_track_terminates_immediately_on_first_miss():
    """Tentative track that is missed before reaching min_hits terminates immediately (flicker noise)."""
    tracker = MultiFaceTracker(min_hits_to_confirm=3)

    tracker.update([_det(1, 100, 100)], frame_seq=1, timestamp=1.0)
    # Missed on Frame 2
    res2 = tracker.update([], frame_seq=2, timestamp=1.1)
    assert res2.tracks[0].state == TrackState.TERMINATED.value


# ==============================================================================
# 4. CROSSING & OVERLAPPING DETECTIONS (AMBIGUITY DETECTION)
# ==============================================================================

def test_crossing_faces_ambiguity_flagging():
    """Requirement 4: When 2 tracks overlap significantly during crossing, flag assignment as ambiguous."""
    tracker = MultiFaceTracker(min_hits_to_confirm=2, ambiguity_iou_threshold=0.20)

    # Frame 1: Track A at (100, 100, 60, 60), Track B at (180, 100, 60, 60)
    tracker.update([_det(1, 100, 100, 60, 60), _det(2, 180, 100, 60, 60)], frame_seq=1, timestamp=1.0)
    # Frame 2: Confirmed
    tracker.update([_det(1, 110, 100, 60, 60), _det(2, 170, 100, 60, 60)], frame_seq=2, timestamp=1.1)

    # Frame 3: Crossing! Detections heavily overlap both predicted boxes
    # Det at (135, 100, 60, 60) overlaps both Track A (~120) and Track B (~160)
    res3 = tracker.update([_det(1, 135, 100, 60, 60), _det(2, 145, 100, 60, 60)], frame_seq=3, timestamp=1.2)

    # At least one of the crossing tracks must be flagged as ambiguous
    ambiguous_tracks = [t for t in res3.tracks if t.is_ambiguous]
    assert len(ambiguous_tracks) > 0


# ==============================================================================
# 5. FRAME ORDERING & DUPLICATE / OUT-OF-ORDER REJECTION
# ==============================================================================

def test_reject_duplicate_and_out_of_order_frame_seq():
    """Requirement 7: Duplicate and out-of-order frame sequences must be rejected with ValueError."""
    tracker = MultiFaceTracker()

    tracker.update([_det(1, 100, 100)], frame_seq=5, timestamp=10.0)

    # Duplicate frame_seq 5
    with pytest.raises(ValueError) as exc_dup:
        tracker.update([_det(1, 100, 100)], frame_seq=5, timestamp=10.1)
    assert "Out-of-order or duplicate" in str(exc_dup.value)

    # Decreasing frame_seq 4
    with pytest.raises(ValueError) as exc_retro:
        tracker.update([_det(1, 100, 100)], frame_seq=4, timestamp=10.2)
    assert "Out-of-order or duplicate" in str(exc_retro.value)


def test_reject_negative_time_deltas():
    """Timestamps must not move backward in time."""
    tracker = MultiFaceTracker()
    tracker.update([_det(1, 100, 100)], frame_seq=1, timestamp=10.0)

    with pytest.raises(ValueError) as exc:
        tracker.update([_det(1, 100, 100)], frame_seq=2, timestamp=9.5)
    assert "Negative time delta" in str(exc.value)


# ==============================================================================
# 6. INSTANCE ISOLATION & UNIQUE TRACK ID GUARANTEES
# ==============================================================================

def test_independent_tracker_instances_isolation():
    """Two tracker instances must operate with completely isolated track IDs and states."""
    tracker_A = MultiFaceTracker()
    tracker_B = MultiFaceTracker()

    res_A = tracker_A.update([_det(1, 100, 100)], frame_seq=1, timestamp=1.0)
    res_B = tracker_B.update([_det(1, 500, 500)], frame_seq=1, timestamp=1.0)

    assert res_A.tracks[0].track_id == "TRK-0001"
    assert res_B.tracks[0].track_id == "TRK-0001"  # Starts at 0001 independently

    res_A2 = tracker_A.update([_det(1, 100, 100), _det(2, 200, 200)], frame_seq=2, timestamp=1.1)
    assert res_A2.active_tracks_count == 2
    assert tracker_B.tracks["TRK-0001"].total_observations == 1  # Unaffected


def test_terminated_track_id_never_reassigned():
    """Terminated tracks are discarded; new tracks receive strictly incrementing new IDs."""
    tracker = MultiFaceTracker(min_hits_to_confirm=1, max_missing_frames=1)

    # Track 1 created
    tracker.update([_det(1, 100, 100)], frame_seq=1, timestamp=1.0)
    # Track 1 lost and terminated
    tracker.update([], frame_seq=2, timestamp=1.1)
    tracker.update([], frame_seq=3, timestamp=1.2)

    # Next detection must NOT reuse TRK-0001
    res4 = tracker.update([_det(2, 300, 300)], frame_seq=4, timestamp=1.3)
    assert res4.tracks[0].track_id == "TRK-0002"
    assert "TRK-0001" not in tracker.tracks


def test_invalid_and_zero_size_bounding_boxes_sanitized():
    """Zero-width or negative-coordinate detections must be handled safely without crashing."""
    tracker = MultiFaceTracker()
    invalid_dets = [
        DetectionInput(face_id=1, bounding_box=(10, 10, 0, 50), confidence=0.9),  # 0 width
        DetectionInput(face_id=2, bounding_box=(20, 20, 50, -5), confidence=0.9),  # negative height
        DetectionInput(face_id=3, bounding_box=(50, 50, 40, 40), confidence=0.9),  # valid
    ]

    res = tracker.update(invalid_dets, frame_seq=1, timestamp=1.0)
    # Only 1 valid track created
    assert res.active_tracks_count == 1
    assert res.tracks[0].bounding_box == (50, 50, 40, 40)

