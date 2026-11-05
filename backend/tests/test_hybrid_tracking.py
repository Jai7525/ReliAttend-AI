"""
ReliAttend AI — Substage 9.5 Hybrid Association Automated Tests
Validates:
1. Pure spatial association continues to function reliably.
2. Appearance cosine similarity recovers associations when IoU is degraded (e.g. fast motion).
3. Appearance mismatches (< 0.60 threshold) are gated and rejected.
4. Competing tracks within ambiguity margin produce is_ambiguous=True and do not swap IDs.
5. Poor quality / blurred detections do not contaminate appearance history.
6. Bounded history never exceeds 3 templates per track.
7. End-to-end performance latency benchmark for 1-face and 3-face workloads.
"""

import time
import numpy as np
import pytest

from app.ai.tracking.association import (
    associate_detections_hybrid,
    associate_detections_to_tracks,
    compute_appearance_similarity_matrix,
    compute_iou,
)
from app.ai.tracking.track import FaceTrack, TrackState
from app.ai.tracking.tracker import DetectionInput, MultiFaceTracker


def _random_normalized_embedding(seed: int = 42) -> np.ndarray:
    """Produces deterministic 128-d L2-normalized float32 embedding."""
    rng = np.random.RandomState(seed)
    v = rng.randn(128).astype(np.float32)
    return v / np.linalg.norm(v)


def test_spatial_association_baseline():
    """Confirms traditional high-overlap spatial matching works identically."""
    boxes_a = [(100, 100, 60, 60)]
    boxes_b = [(104, 102, 60, 60)]
    iou = compute_iou(boxes_a[0], boxes_b[0])
    assert iou > 0.80

    matches, unmatched_t, unmatched_d = associate_detections_to_tracks(
        boxes_a, boxes_b, iou_threshold=0.30
    )
    assert matches == [(0, 0)]
    assert len(unmatched_t) == 0
    assert len(unmatched_d) == 0


def test_appearance_recovers_low_iou_match():
    """
    Simulates rapid head motion where IoU drops to 0.15 (below iou_threshold=0.30).
    With matching SFace appearance (cosine similarity > 0.90), hybrid association
    successfully associates the detection to the track without spawning a new ID.
    """
    track_boxes = [(100, 100, 60, 60)]
    # Shifted by 45 pixels: IoU is approximately 0.15 (below 0.30 threshold)
    det_boxes = [(145, 100, 60, 60)]
    iou = compute_iou(track_boxes[0], det_boxes[0])
    assert 0.05 < iou < 0.30

    emb_person1 = _random_normalized_embedding(100)
    track_embs = [[emb_person1]]
    det_embs = [emb_person1]

    # Pure spatial association would FAIL here:
    matches_spatial, _, _ = associate_detections_to_tracks(track_boxes, det_boxes, iou_threshold=0.30)
    assert len(matches_spatial) == 0

    # Hybrid association SUCCEEDS via appearance similarity:
    matches, unmatched_t, unmatched_d, ambiguous = associate_detections_hybrid(
        track_boxes=track_boxes,
        detection_boxes=det_boxes,
        track_embeddings=track_embs,
        detection_embeddings=det_embs,
        iou_threshold=0.30,
        appearance_threshold=0.60,
    )
    assert matches == [(0, 0)]
    assert len(unmatched_t) == 0
    assert len(unmatched_d) == 0
    assert len(ambiguous) == 0


def test_appearance_mismatch_rejected():
    """
    If IoU is below 0.30 and appearance does not match (different person),
    association is rejected and detection is marked unmatched.
    """
    track_boxes = [(100, 100, 60, 60)]
    det_boxes = [(145, 100, 60, 60)]

    emb_person1 = _random_normalized_embedding(101)
    emb_person2 = _random_normalized_embedding(202)  # Orthogonal / different person
    sim = float(np.dot(emb_person1, emb_person2))
    assert sim < 0.50

    matches, unmatched_t, unmatched_d, _ = associate_detections_hybrid(
        track_boxes=track_boxes,
        detection_boxes=det_boxes,
        track_embeddings=[[emb_person1]],
        detection_embeddings=[emb_person2],
        iou_threshold=0.30,
        appearance_threshold=0.60,
    )
    assert len(matches) == 0
    assert unmatched_t == [0]
    assert unmatched_d == [0]


def test_competing_tracks_produce_ambiguity():
    """
    When two tracks both have significant spatial overlap or close appearance scores
    with a single detection (crossing scenario), ambiguity flag is asserted.
    """
    # Two tracks crossing at same coordinates
    track_boxes = [(100, 100, 60, 60), (105, 102, 60, 60)]
    det_boxes = [(102, 101, 60, 60)]

    emb1 = _random_normalized_embedding(1)
    emb2 = _random_normalized_embedding(2)

    matches, _, _, ambiguous = associate_detections_hybrid(
        track_boxes=track_boxes,
        detection_boxes=det_boxes,
        track_embeddings=[[emb1], [emb2]],
        detection_embeddings=[emb1],
        iou_threshold=0.30,
        ambiguity_margin=0.08,
    )
    assert len(matches) == 1
    # Match index 0 must be flagged as ambiguous
    assert 0 in ambiguous


def test_bounded_appearance_history_size():
    """Confirms appearance template deque never exceeds 3 entries."""
    track = FaceTrack(
        track_id="TRK-0001",
        initial_bbox=(100, 100, 50, 50),
        frame_seq=0,
        timestamp=100.0,
        detection_confidence=0.90,
    )

    for i in range(5):
        emb = _random_normalized_embedding(i)
        track.add_appearance_embedding(emb, is_quality_acceptable=True, is_ambiguous=False)

    templates = track.get_appearance_embeddings()
    assert len(templates) == 3
    # Most recent entries should be preserved
    assert np.allclose(templates[-1], _random_normalized_embedding(4))


def test_poor_quality_and_ambiguous_detections_do_not_contaminate_history():
    """Poor quality or ambiguous detections must be rejected from template history."""
    track = FaceTrack(
        track_id="TRK-0001",
        initial_bbox=(100, 100, 50, 50),
        frame_seq=0,
        timestamp=100.0,
        detection_confidence=0.90,
    )

    valid_emb = _random_normalized_embedding(10)
    # 1. Accepted when valid
    assert track.add_appearance_embedding(valid_emb, is_quality_acceptable=True, is_ambiguous=False) is True
    assert len(track.get_appearance_embeddings()) == 1

    # 2. Rejected when quality unacceptable
    low_q_emb = _random_normalized_embedding(20)
    assert track.add_appearance_embedding(low_q_emb, is_quality_acceptable=False, is_ambiguous=False) is False
    assert len(track.get_appearance_embeddings()) == 1

    # 3. Rejected when ambiguous
    amb_emb = _random_normalized_embedding(30)
    assert track.add_appearance_embedding(amb_emb, is_quality_acceptable=True, is_ambiguous=True) is False
    assert len(track.get_appearance_embeddings()) == 1


def test_tracker_end_to_end_straightforward_motion():
    """Tests multi-frame tracking sequence with appearance features across 10 frames."""
    tracker = MultiFaceTracker(min_hits_to_confirm=3)
    emb = _random_normalized_embedding(77)

    for f in range(10):
        # Moving by 10 pixels per frame
        bbox = (100 + f * 10, 100, 60, 60)
        det = DetectionInput(
            face_id=1,
            bounding_box=bbox,
            confidence=0.95,
            embedding=emb,
            is_quality_acceptable=True,
        )
        res = tracker.update([det], frame_seq=f, timestamp=100.0 + f * 0.2)

        assert res.active_tracks_count == 1
        assert res.tracks[0].track_id == "TRK-0001"
        if f >= 2:
            assert res.tracks[0].state == "confirmed"


def test_hybrid_association_latency_benchmark():
    """
    Benchmarks actual association computation time for 1-face and 3-face scenarios.
    Must comfortably execute well below 10 ms (budget for 5 FPS is 200 ms).
    """
    # 1 Face
    track_boxes_1 = [(100, 100, 60, 60)]
    det_boxes_1 = [(110, 105, 60, 60)]
    embs_1 = [_random_normalized_embedding(1)]

    t0 = time.perf_counter()
    for _ in range(50):
        associate_detections_hybrid(
            track_boxes_1, det_boxes_1, [embs_1], embs_1
        )
    t1 = time.perf_counter()
    latency_1face_ms = ((t1 - t0) / 50) * 1000.0

    # 3 Faces
    track_boxes_3 = [(100, 100, 60, 60), (300, 100, 60, 60), (500, 100, 60, 60)]
    det_boxes_3 = [(105, 102, 60, 60), (308, 98, 60, 60), (495, 102, 60, 60)]
    embs_3 = [_random_normalized_embedding(i) for i in range(3)]

    t2 = time.perf_counter()
    for _ in range(50):
        associate_detections_hybrid(
            track_boxes_3, det_boxes_3, [[e] for e in embs_3], embs_3
        )
    t3 = time.perf_counter()
    latency_3face_ms = ((t3 - t2) / 50) * 1000.0

    print(f"\n[LATENCY BENCHMARK] 1 Face Association: {latency_1face_ms:.3f} ms | 3 Faces Association: {latency_3face_ms:.3f} ms")
    assert latency_1face_ms < 5.0
    assert latency_3face_ms < 10.0

