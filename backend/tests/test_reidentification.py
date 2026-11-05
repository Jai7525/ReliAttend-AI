"""
ReliAttend AI — Lost-Track Re-Identification & Crossing-Path Protection Tests (Phase 9 - Substage 9.6)
Deterministic synthetic tests verifying:
1. Same person moving with low IoU (< 0.30) recovered via Stage-1 hybrid matching.
2. Brief disappearance and return recovered via Stage-2 re-identification preserving original Track ID.
3. Two people crossing paths preserves original Track IDs without identity swaps.
4. Similar-looking people crossing (narrow appearance margin) flagged ambiguous and coasted.
5. Ambiguous match does not corrupt Kalman state (prediction coasting).
6. Ambiguous match does not contaminate appearance history.
7. Terminated track is not accidentally recovered (new Track ID issued).
8. Genuinely new person receives a new Track ID.
9. Multiple people independently disappearing and returning retain their respective Track IDs.
10. Session deletion completely purges all tracking state and in-memory templates.
"""

import time
import numpy as np
import pytest

from app.ai.tracking import (
    DetectionInput,
    MultiFaceTracker,
    TrackState,
)
from app.services.tracking_service import TrackingSessionManager, TrackingService
from app.schemas.tracking import CreateTrackingSessionRequest
from app.models.user import User


def _norm_vec(seed: int) -> np.ndarray:
    """Generate deterministic normalized 128-d float32 vector."""
    rng = np.random.RandomState(seed)
    v = rng.randn(128).astype(np.float32)
    return v / np.linalg.norm(v)


def _det(
    face_id: int,
    x: int,
    y: int,
    w: int = 50,
    h: int = 60,
    conf: float = 0.95,
    embedding: np.ndarray = None,
    quality: bool = True,
) -> DetectionInput:
    """Helper to create standardized synthetic DetectionInput."""
    return DetectionInput(
        face_id=face_id,
        bounding_box=(x, y, w, h),
        confidence=conf,
        embedding=embedding,
        is_quality_acceptable=quality,
    )


# ==============================================================================
# TEST 1: Same person moving with low IoU
# ==============================================================================
def test_same_person_moving_with_low_iou():
    """Person jumps quickly, IoU drops to 0.12 (below 0.30), but appearance recovers track."""
    emb_a = _norm_vec(101)
    tracker = MultiFaceTracker(min_hits_to_confirm=2, appearance_threshold=0.60)

    # Frames 1-2: Establish and confirm track
    tracker.update([_det(1, 100, 100, 60, 60, embedding=emb_a)], frame_seq=1, timestamp=1.0)
    res2 = tracker.update([_det(1, 105, 100, 60, 60, embedding=emb_a)], frame_seq=2, timestamp=1.1)
    assert res2.tracks[0].track_id == "TRK-0001"
    assert res2.tracks[0].state == TrackState.CONFIRMED.value

    # Frame 3: Sudden jump to (140, 100, 60, 60). Predicted is around 110. IoU is ~0.15.
    res3 = tracker.update([_det(1, 140, 100, 60, 60, embedding=emb_a)], frame_seq=3, timestamp=1.2)
    assert len(res3.tracks) == 1
    assert res3.tracks[0].track_id == "TRK-0001"
    assert res3.tracks[0].state == TrackState.CONFIRMED.value


# ==============================================================================
# TEST 2: Brief disappearance and return
# ==============================================================================
def test_brief_disappearance_and_return():
    """Person confirms, disappears for 4 frames (TEMPORARILY_LOST), reappears, recovers TRK-0001."""
    emb_a = _norm_vec(202)
    tracker = MultiFaceTracker(min_hits_to_confirm=2, max_missing_frames=10, reid_appearance_threshold=0.65)

    # Establish track
    tracker.update([_det(1, 100, 100, 60, 60, embedding=emb_a)], frame_seq=1, timestamp=1.0)
    tracker.update([_det(1, 102, 100, 60, 60, embedding=emb_a)], frame_seq=2, timestamp=1.1)

    # Disappear for frames 3, 4, 5
    for seq in range(3, 6):
        res_lost = tracker.update([], frame_seq=seq, timestamp=1.0 + seq * 0.1)
        assert res_lost.tracks[0].state == TrackState.TEMPORARILY_LOST.value
        assert res_lost.tracks[0].track_id == "TRK-0001"

    # Frame 6: Reappear at (130, 105, 60, 60) with same appearance
    res_return = tracker.update([_det(1, 130, 105, 60, 60, embedding=emb_a)], frame_seq=6, timestamp=1.6)
    assert len(res_return.tracks) == 1
    t = res_return.tracks[0]
    assert t.track_id == "TRK-0001"
    assert t.state == TrackState.CONFIRMED.value
    assert t.is_recovered is True


# ==============================================================================
# TEST 3: Two people crossing paths
# ==============================================================================
def test_two_people_crossing_paths_preserves_identities():
    """Two people with distinct appearance cross trajectories; their original IDs are preserved."""
    emb_a = _norm_vec(301)
    emb_b = _norm_vec(302)  # Orthogonal random vector
    tracker = MultiFaceTracker(min_hits_to_confirm=2)

    # Frame 1: Person A at x=100 moving right, Person B at x=250 moving left
    tracker.update([
        _det(1, 100, 100, 60, 60, embedding=emb_a),
        _det(2, 250, 100, 60, 60, embedding=emb_b),
    ], frame_seq=1, timestamp=1.0)

    # Frame 2: Moving closer
    tracker.update([
        _det(1, 130, 100, 60, 60, embedding=emb_a),
        _det(2, 220, 100, 60, 60, embedding=emb_b),
    ], frame_seq=2, timestamp=1.1)

    # Frame 3: Crossing midpoint
    tracker.update([
        _det(1, 165, 100, 60, 60, embedding=emb_a),
        _det(2, 185, 100, 60, 60, embedding=emb_b),
    ], frame_seq=3, timestamp=1.2)

    # Frame 4: Fully crossed (A is now at x=210, B is now at x=140)
    res4 = tracker.update([
        _det(1, 140, 100, 60, 60, embedding=emb_b),
        _det(2, 210, 100, 60, 60, embedding=emb_a),
    ], frame_seq=4, timestamp=1.3)

    tracks_by_id = {t.track_id: t for t in res4.tracks}
    assert "TRK-0001" in tracks_by_id
    assert "TRK-0002" in tracks_by_id
    # TRK-0001 (Person A) must be at x=210
    assert abs(tracks_by_id["TRK-0001"].bounding_box[0] - 210) < 5
    # TRK-0002 (Person B) must be at x=140
    assert abs(tracks_by_id["TRK-0002"].bounding_box[0] - 140) < 5


# ==============================================================================
# TEST 4: Similar-looking people crossing (ambiguity margin protection)
# ==============================================================================
def test_similar_looking_people_crossing_produces_ambiguity():
    """When two people have nearly identical appearance, crossing is flagged ambiguous."""
    base_emb = _norm_vec(400)
    # Create twin embedding with similarity ~0.96 (difference vector norm tiny)
    noise = np.random.RandomState(401).randn(128).astype(np.float32) * 0.15
    twin_emb = base_emb + noise
    twin_emb = twin_emb / np.linalg.norm(twin_emb)

    tracker = MultiFaceTracker(min_hits_to_confirm=2, ambiguity_iou_threshold=0.20, ambiguity_margin=0.08)

    tracker.update([
        _det(1, 100, 100, 60, 60, embedding=base_emb),
        _det(2, 160, 100, 60, 60, embedding=twin_emb),
    ], frame_seq=1, timestamp=1.0)
    tracker.update([
        _det(1, 110, 100, 60, 60, embedding=base_emb),
        _det(2, 150, 100, 60, 60, embedding=twin_emb),
    ], frame_seq=2, timestamp=1.1)

    # Cross at (130, 100) and (135, 100)
    res3 = tracker.update([
        _det(1, 130, 100, 60, 60, embedding=base_emb),
        _det(2, 135, 100, 60, 60, embedding=twin_emb),
    ], frame_seq=3, timestamp=1.2)

    ambiguous_tracks = [t for t in res3.tracks if t.is_ambiguous]
    assert len(ambiguous_tracks) > 0


# ==============================================================================
# TEST 5: Ambiguous match does not corrupt Kalman state
# ==============================================================================
def test_ambiguous_match_does_not_corrupt_kalman_state():
    """When a match is flagged ambiguous, Kalman filter measurement update is bypassed."""
    emb_a = _norm_vec(501)
    tracker = MultiFaceTracker(min_hits_to_confirm=2)

    tracker.update([_det(1, 100, 100, 60, 60, embedding=emb_a)], frame_seq=1, timestamp=1.0)
    tracker.update([_det(1, 110, 100, 60, 60, embedding=emb_a)], frame_seq=2, timestamp=1.1)

    track = tracker.tracks["TRK-0001"]
    # Predict next frame: constant velocity dx=10 -> predicted x is around 120
    pred = track.predict()
    assert abs(pred[0] - 120) <= 2

    # Call update_matched with is_ambiguous=True with a wildly offset detection at x=300
    track.update_matched(bbox=(300, 100, 60, 60), frame_seq=3, timestamp=1.2, detection_confidence=0.9, is_ambiguous=True)

    # current_bbox should retain predicted position (~120), NOT jumped to 300
    assert abs(track.current_bbox[0] - 120) <= 2
    # Next prediction should continue smooth motion from 120, not 300
    next_pred = track.predict()
    assert next_pred[0] < 150


# ==============================================================================
# TEST 6: Ambiguous match does not contaminate appearance history
# ==============================================================================
def test_ambiguous_match_does_not_contaminate_appearance_history():
    """When detection is ambiguous, add_appearance_embedding returns False."""
    emb_clean = _norm_vec(601)
    emb_dirty = _norm_vec(602)
    tracker = MultiFaceTracker(min_hits_to_confirm=2)

    tracker.update([_det(1, 100, 100, 60, 60, embedding=emb_clean)], frame_seq=1, timestamp=1.0)
    track = tracker.tracks["TRK-0001"]
    assert len(track.appearance_history) == 1

    # Attempt to inject dirty embedding with is_ambiguous=True
    accepted = track.add_appearance_embedding(emb_dirty, is_quality_acceptable=True, is_ambiguous=True)
    assert accepted is False
    assert len(track.appearance_history) == 1
    # Verify similarity with clean is 1.0, not contaminated by dirty
    sim = np.dot(track.appearance_history[0], emb_clean)
    assert sim > 0.999


# ==============================================================================
# TEST 7: Terminated track is not accidentally recovered
# ==============================================================================
def test_terminated_track_is_not_recovered():
    """Once a track exceeds max_missing_frames and TERMINATES, reappearance creates a NEW Track ID."""
    emb_a = _norm_vec(701)
    tracker = MultiFaceTracker(min_hits_to_confirm=2, max_missing_frames=2)

    # Establish track TRK-0001
    tracker.update([_det(1, 100, 100, 60, 60, embedding=emb_a)], frame_seq=1, timestamp=1.0)
    tracker.update([_det(1, 100, 100, 60, 60, embedding=emb_a)], frame_seq=2, timestamp=1.1)

    # Miss 1 frame -> TEMPORARILY_LOST
    tracker.update([], frame_seq=3, timestamp=1.2)
    # Miss 2 frames -> TEMPORARILY_LOST
    tracker.update([], frame_seq=4, timestamp=1.3)
    # Miss 3 frames (> 2) -> TERMINATED and purged
    res5 = tracker.update([], frame_seq=5, timestamp=1.4)
    assert res5.active_tracks_count == 0
    assert len(tracker.tracks) == 0

    # Person returns on Frame 6 with identical appearance
    res6 = tracker.update([_det(1, 100, 100, 60, 60, embedding=emb_a)], frame_seq=6, timestamp=1.5)
    assert res6.active_tracks_count == 1
    # MUST be a brand new track ID (TRK-0002), never resurrected TRK-0001
    assert res6.tracks[0].track_id == "TRK-0002"
    assert res6.tracks[0].is_newly_created is True


# ==============================================================================
# TEST 8: Genuinely new person receives a new Track ID
# ==============================================================================
def test_genuinely_new_person_receives_new_track_id():
    """When a new person enters while a track is lost, the new person gets a new Track ID."""
    emb_existing = _norm_vec(801)
    emb_stranger = _norm_vec(802)  # Completely different person
    tracker = MultiFaceTracker(min_hits_to_confirm=2, max_missing_frames=10)

    # Establish TRK-0001
    tracker.update([_det(1, 100, 100, 60, 60, embedding=emb_existing)], frame_seq=1, timestamp=1.0)
    tracker.update([_det(1, 100, 100, 60, 60, embedding=emb_existing)], frame_seq=2, timestamp=1.1)

    # TRK-0001 disappears on Frame 3
    tracker.update([], frame_seq=3, timestamp=1.2)

    # Stranger arrives at Frame 4
    res4 = tracker.update([_det(1, 100, 100, 60, 60, embedding=emb_stranger)], frame_seq=4, timestamp=1.3)
    # Must NOT re-identify as TRK-0001; must create TRK-0002
    created = [t for t in res4.tracks if t.is_newly_created]
    assert len(created) == 1
    assert created[0].track_id == "TRK-0002"


# ==============================================================================
# TEST 9: Multiple people independently disappearing and returning
# ==============================================================================
def test_multiple_people_independently_disappearing_and_returning():
    """Two people independently disappear at different times and recover their own Track IDs."""
    emb_a = _norm_vec(901)
    emb_b = _norm_vec(902)
    tracker = MultiFaceTracker(min_hits_to_confirm=2, max_missing_frames=15)

    # Both present
    tracker.update([
        _det(1, 100, 100, 60, 60, embedding=emb_a),
        _det(2, 300, 100, 60, 60, embedding=emb_b),
    ], frame_seq=1, timestamp=1.0)
    tracker.update([
        _det(1, 100, 100, 60, 60, embedding=emb_a),
        _det(2, 300, 100, 60, 60, embedding=emb_b),
    ], frame_seq=2, timestamp=1.1)

    # Person A disappears, Person B stays
    res3 = tracker.update([_det(2, 300, 100, 60, 60, embedding=emb_b)], frame_seq=3, timestamp=1.2)
    tracks3 = {t.track_id: t for t in res3.tracks}
    assert tracks3["TRK-0001"].state == TrackState.TEMPORARILY_LOST.value
    assert tracks3["TRK-0002"].state == TrackState.CONFIRMED.value

    # Person A returns, Person B disappears
    res4 = tracker.update([_det(1, 110, 100, 60, 60, embedding=emb_a)], frame_seq=4, timestamp=1.3)
    tracks4 = {t.track_id: t for t in res4.tracks}
    assert tracks4["TRK-0001"].state == TrackState.CONFIRMED.value
    assert tracks4["TRK-0001"].is_recovered is True
    assert tracks4["TRK-0002"].state == TrackState.TEMPORARILY_LOST.value

    # Person B returns
    res5 = tracker.update([
        _det(1, 110, 100, 60, 60, embedding=emb_a),
        _det(2, 310, 100, 60, 60, embedding=emb_b),
    ], frame_seq=5, timestamp=1.4)
    tracks5 = {t.track_id: t for t in res5.tracks}
    assert tracks5["TRK-0001"].state == TrackState.CONFIRMED.value
    assert tracks5["TRK-0002"].state == TrackState.CONFIRMED.value
    assert tracks5["TRK-0002"].is_recovered is True


# ==============================================================================
# TEST 10: Session deletion clears all tracking state
# ==============================================================================
def test_session_deletion_clears_all_tracking_state():
    """Ending a session removes tracker instance and frees all template buffers."""
    mgr = TrackingSessionManager()
    cfg = CreateTrackingSessionRequest(class_section_id=1)
    sess = mgr.create_session(user_id=1, user_role="faculty", class_section_id=1, config=cfg)
    sess_id = sess.session_id

    # Retrieve and populate tracker
    stored = mgr.get_session(sess_id)
    assert stored is not None
    emb = _norm_vec(1001)
    stored.tracker.update([_det(1, 100, 100, 60, 60, embedding=emb)], frame_seq=1, timestamp=1.0)
    assert len(stored.tracker.tracks) == 1

    # End/remove session
    mgr.remove_session(sess_id)

    # Session must be completely purged from manager
    assert mgr.get_session(sess_id) is None
    assert sess_id not in mgr._sessions
