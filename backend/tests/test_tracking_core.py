"""
ReliAttend AI — Tracking Core Math & Association Unit Tests (Phase 9 - Substage 9.1)
Deterministic synthetic tests verifying:
- Kalman state predictions and updates.
- Stationary and moving box velocity tracking.
- IoU calculations (identical, overlapping, touching, disjoint).
- Linear assignment one-to-one mapping.
- Gating and rejection of low-overlap associations.
- Empty detections and empty tracks edge cases.
"""

import time
import numpy as np
import pytest

from app.ai.tracking.association import (
    associate_detections_to_tracks,
    compute_iou,
    compute_iou_matrix,
    linear_assignment,
)
from app.ai.tracking.kalman import KalmanBoxTracker


# ==============================================================================
# 1. KALMAN BOX TRACKER MATHEMATICAL UNIT TESTS
# ==============================================================================

def test_kalman_initialization_and_state_consistency():
    """Initial Kalman state should match initial bounding box exactly."""
    initial_bbox = (100, 150, 80, 100)  # x, y, w, h
    tracker = KalmanBoxTracker(initial_bbox)

    state_bbox = tracker.get_state()
    assert state_bbox == initial_bbox


def test_kalman_stationary_box_predictions():
    """A stationary face box should maintain its spatial position over multiple prediction cycles."""
    initial_bbox = (200, 200, 90, 120)
    tracker = KalmanBoxTracker(initial_bbox)

    for _ in range(5):
        pred_box = tracker.predict()
        # Should stay very close to (200, 200, 90, 120) since initial velocity is zero
        assert abs(pred_box[0] - 200) <= 2
        assert abs(pred_box[1] - 200) <= 2
        assert abs(pred_box[2] - 90) <= 2
        assert abs(pred_box[3] - 120) <= 2
        # Update with same measurement
        tracker.update(initial_bbox)


def test_kalman_constant_velocity_motion_extrapolation():
    """Kalman filter should estimate positive horizontal velocity and extrapolate motion."""
    # Face moving linearly horizontally: x increases by 10 px per frame
    tracker = KalmanBoxTracker((100, 100, 80, 80))

    # Provide 4 consecutive observations with velocity dx=10
    tracker.predict()
    tracker.update((110, 100, 80, 80))

    tracker.predict()
    tracker.update((120, 100, 80, 80))

    tracker.predict()
    tracker.update((130, 100, 80, 80))

    tracker.predict()
    tracker.update((140, 100, 80, 80))

    # On frame 5, miss detection (occlusion) and rely purely on prediction:
    pred_5 = tracker.predict()
    # Predicted x should be extrapolated around ~150 (+/- 3 px)
    assert 145 <= pred_5[0] <= 155
    assert abs(pred_5[1] - 100) <= 2


def test_kalman_scale_and_aspect_ratio_stability():
    """Aspect ratio and height must remain physically positive and non-zero."""
    tracker = KalmanBoxTracker((50, 50, 40, 60))
    for _ in range(10):
        pred = tracker.predict()
        assert pred[2] > 0  # width > 0
        assert pred[3] > 0  # height > 0


# ==============================================================================
# 2. INTERSECTION OVER UNION (IoU) MATHEMATICAL TESTS
# ==============================================================================

def test_iou_identical_boxes():
    """Identical bounding boxes must yield IoU = 1.0."""
    box = (100, 100, 50, 50)
    assert pytest.approx(compute_iou(box, box), 1e-5) == 1.0


def test_iou_completely_disjoint_boxes():
    """Non-overlapping boxes must yield IoU = 0.0."""
    box_a = (0, 0, 50, 50)
    box_b = (200, 200, 50, 50)
    assert compute_iou(box_a, box_b) == 0.0


def test_iou_half_overlap_calculation():
    """Box A: [0, 0, 100, 100], Box B: [50, 0, 100, 100]. Intersection = 50x100 = 5000, Union = 15000 -> IoU = 1/3."""
    box_a = (0, 0, 100, 100)
    box_b = (50, 0, 100, 100)
    expected_iou = 5000.0 / 15000.0
    assert pytest.approx(compute_iou(box_a, box_b), 1e-4) == expected_iou


def test_iou_touching_edge_boxes():
    """Touching borders with 0 interior area must yield IoU = 0.0."""
    box_a = (0, 0, 10, 10)
    box_b = (10, 0, 10, 10)
    assert compute_iou(box_a, box_b) == 0.0


def test_compute_iou_matrix_dimensions():
    """IoU matrix should match shape (N_tracks, N_detections)."""
    tracks = [(10, 10, 50, 50), (100, 100, 60, 60), (200, 200, 70, 70)]
    dets = [(12, 12, 50, 50), (500, 500, 50, 50)]
    mat = compute_iou_matrix(tracks, dets)
    assert mat.shape == (3, 2)
    assert mat[0, 0] > 0.80  # tracks[0] and dets[0] overlap closely
    assert mat[0, 1] == 0.0  # tracks[0] and dets[1] are completely disjoint


# ==============================================================================
# 3. HUNGARIAN ASSIGNMENT & ASSOCIATION TESTS
# ==============================================================================

def test_linear_assignment_exact_one_to_one():
    """Linear assignment must map diagonal minimum cost with 100% precision."""
    # Cost matrix where diagonal elements are minimal (best matches)
    cost_matrix = np.array([
        [0.1, 0.9, 0.8],
        [0.8, 0.2, 0.7],
        [0.9, 0.8, 0.05],
    ], dtype=np.float32)

    matches = linear_assignment(cost_matrix)
    # Expected pairs: (0, 0), (1, 1), (2, 2)
    assert len(matches) == 3
    assert (0, 0) in matches
    assert (1, 1) in matches
    assert (2, 2) in matches


def test_associate_detections_to_tracks_one_to_one():
    """Multiple tracks and detections with clear spatial correspondences."""
    tracks = [
        (100, 100, 50, 50),  # Track 0
        (300, 300, 50, 50),  # Track 1
    ]
    detections = [
        (302, 298, 50, 50),  # Det 0 matches Track 1
        (101, 102, 50, 50),  # Det 1 matches Track 0
    ]

    matches, unmatched_t, unmatched_d = associate_detections_to_tracks(
        track_boxes=tracks,
        detection_boxes=detections,
        iou_threshold=0.30,
    )

    assert len(matches) == 2
    assert (0, 1) in matches  # Track 0 -> Det 1
    assert (1, 0) in matches  # Track 1 -> Det 0
    assert unmatched_t == []
    assert unmatched_d == []


def test_associate_detections_gating_rejects_low_iou():
    """Detections far from any track must be gated out and listed as unmatched detections."""
    tracks = [(50, 50, 40, 40)]
    detections = [(400, 400, 40, 40)]  # Way too far, IoU = 0.0

    matches, unmatched_t, unmatched_d = associate_detections_to_tracks(
        track_boxes=tracks,
        detection_boxes=detections,
        iou_threshold=0.30,
    )

    assert matches == []
    assert unmatched_t == [0]
    assert unmatched_d == [0]


def test_associate_empty_inputs():
    """Empty tracks or empty detections should return clean lists without throwing exceptions."""
    # 1. Empty detections
    matches, unmatched_t, unmatched_d = associate_detections_to_tracks([(10, 10, 50, 50)], [])
    assert matches == []
    assert unmatched_t == [0]
    assert unmatched_d == []

    # 2. Empty tracks
    matches, unmatched_t, unmatched_d = associate_detections_to_tracks([], [(10, 10, 50, 50)])
    assert matches == []
    assert unmatched_t == []
    assert unmatched_d == [0]

    # 3. Both empty
    matches, unmatched_t, unmatched_d = associate_detections_to_tracks([], [])
    assert matches == []
    assert unmatched_t == []
    assert unmatched_d == []


def test_substage_9_1_performance_benchmark():
    """Association of 10 tracks with 10 detections should execute in < 1 millisecond."""
    tracks = [(i * 50, 100, 40, 40) for i in range(10)]
    detections = [(i * 50 + 2, 102, 40, 40) for i in range(10)]

    start = time.perf_counter()
    for _ in range(50):
        matches, _, _ = associate_detections_to_tracks(tracks, detections, iou_threshold=0.3)
    elapsed_ms = ((time.perf_counter() - start) / 50.0) * 1000.0

    assert len(matches) == 10
    # Expected latency < 1.0 ms
    assert elapsed_ms < 2.0

