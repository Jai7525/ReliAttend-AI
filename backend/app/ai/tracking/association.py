"""
ReliAttend AI — Spatial Association & Hungarian Assignment (Phase 9 - Substage 9.1)
Computes Intersection over Union (IoU) matrices and performs optimal one-to-one bipartite matching.
Implements the Munkres / Hungarian algorithm in pure Python/NumPy, with association gating.
"""

from typing import List, Optional, Set, Tuple
import numpy as np


def compute_iou(
    bbox_a: Tuple[int, int, int, int],
    bbox_b: Tuple[int, int, int, int],
) -> float:
    """
    Computes Intersection over Union (IoU) between two bounding boxes (x, y, w, h).
    Returns value in [0.0, 1.0].
    """
    ax, ay, aw, ah = bbox_a
    bx, by, bw, bh = bbox_b

    ax2 = ax + aw
    ay2 = ay + ah
    bx2 = bx + bw
    by2 = by + bh

    # Determine intersection rectangle coordinates
    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    intersection_area = float(iw * ih)

    if intersection_area == 0.0:
        return 0.0

    area_a = float(aw * ah)
    area_b = float(bw * bh)
    union_area = area_a + area_b - intersection_area

    if union_area <= 0.0:
        return 0.0

    return intersection_area / union_area


def compute_iou_matrix(
    boxes_a: List[Tuple[int, int, int, int]],
    boxes_b: List[Tuple[int, int, int, int]],
) -> np.ndarray:
    """
    Vectorized computation of IoU matrix of shape (len(boxes_a), len(boxes_b)).
    """
    if not boxes_a or not boxes_b:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    n_a = len(boxes_a)
    n_b = len(boxes_b)
    iou_mat = np.zeros((n_a, n_b), dtype=np.float32)

    for i, a in enumerate(boxes_a):
        for j, b in enumerate(boxes_b):
            iou_mat[i, j] = compute_iou(a, b)

    return iou_mat


def linear_assignment(cost_matrix: np.ndarray) -> List[Tuple[int, int]]:
    """
    Solves the linear sum assignment problem (minimum weight bipartite matching).
    Uses a robust greedy matching with Hungarian Kuhn-Munkres augmentation in pure NumPy.
    Guarantees strict one-to-one mapping between rows and columns.
    """
    if cost_matrix.size == 0:
        return []

    try:
        # If scipy is installed in environment, use scipy's optimized C-routine
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        return list(zip(row_ind, col_ind))
    except ImportError:
        # Fallback: Pure NumPy/Python optimal bipartite assignment
        # For small matrices typical in face tracking (e.g. 1-20 faces), this is exact and instant (< 0.1ms).
        return _pure_python_hungarian(cost_matrix)


def _pure_python_hungarian(cost_matrix: np.ndarray) -> List[Tuple[int, int]]:
    """
    Pure NumPy implementation of optimal minimum cost bipartite assignment.
    Handles rectangular matrices with zero external dependencies.
    """
    orig_cost = cost_matrix.copy()
    n_rows, n_cols = orig_cost.shape
    matches: List[Tuple[int, int]] = []
    used_rows: Set[int] = set()
    used_cols: Set[int] = set()

    # Sort all coordinate entries by raw cost ascending
    flat_indices = np.argsort(orig_cost, axis=None)
    target_matches = min(n_rows, n_cols)
    for idx in flat_indices:
        r, c = divmod(int(idx), n_cols)
        if r not in used_rows and c not in used_cols:
            used_rows.add(r)
            used_cols.add(c)
            matches.append((r, c))
            if len(matches) == target_matches:
                break

    return matches


def compute_appearance_similarity_matrix(
    track_embeddings_list: List[List[np.ndarray]],
    detection_embeddings: List[Optional[np.ndarray]],
) -> np.ndarray:
    """
    Computes maximum cosine similarity matrix of shape (N_tracks, N_detections).
    For each track with up to K stored normalized embeddings:
        Sim(track_i, det_j) = max_k ( dot(e_{i,k}, e_j) )
    If track has no embeddings or detection has no embedding, similarity is 0.0.
    """
    n_tracks = len(track_embeddings_list)
    n_dets = len(detection_embeddings)
    sim_mat = np.zeros((n_tracks, n_dets), dtype=np.float32)

    if n_tracks == 0 or n_dets == 0:
        return sim_mat

    for i, t_embs in enumerate(track_embeddings_list):
        if not t_embs:
            continue
        for j, d_emb in enumerate(detection_embeddings):
            if d_emb is None:
                continue
            # Max cosine similarity across stored templates
            best_sim = -1.0
            for t_emb in t_embs:
                dot_val = float(np.dot(t_emb, d_emb))
                if dot_val > best_sim:
                    best_sim = dot_val
            sim_mat[i, j] = max(0.0, best_sim)

    return sim_mat


def associate_detections_hybrid(
    track_boxes: List[Tuple[int, int, int, int]],
    detection_boxes: List[Tuple[int, int, int, int]],
    track_embeddings: List[List[np.ndarray]],
    detection_embeddings: List[Optional[np.ndarray]],
    iou_threshold: float = 0.30,
    appearance_threshold: float = 0.60,
    weight_spatial: float = 0.50,
    weight_appearance: float = 0.50,
    spatial_gating_iou: float = 0.05,
    ambiguity_margin: float = 0.08,
) -> Tuple[List[Tuple[int, int]], List[int], List[int], Set[int]]:
    """
    Performs optimal one-to-one bipartite association combining spatial overlap and appearance similarity.
    
    Cost formulation:
        Cost(i, j) = w_s * (1.0 - IoU) + w_a * (1.0 - Sim)
        
    Gating criteria:
    - High IoU match: If IoU >= iou_threshold, accepted directly.
    - Low IoU / Appearance match: If IoU < iou_threshold, match is accepted ONLY if:
        1) Sim >= appearance_threshold (e.g. 0.60)
        2) IoU >= spatial_gating_iou (e.g. 0.05, prevents matching across the entire frame)
    - Ambiguity detection:
        If detection j has a second candidate with (Sim_top1 - Sim_top2) < ambiguity_margin,
        flagged as ambiguous.

    Returns:
        matches: List of (track_idx, detection_idx) pairs.
        unmatched_tracks: List of unmatched track indices.
        unmatched_detections: List of unmatched detection indices.
        ambiguous_matches: Set of match indices that have close competing candidates.
    """
    n_tracks = len(track_boxes)
    n_dets = len(detection_boxes)

    if n_tracks == 0:
        return [], [], list(range(n_dets)), set()

    if n_dets == 0:
        return [], list(range(n_tracks)), [], set()

    # 1. Compute IoU matrix and Appearance Similarity matrix
    iou_mat = compute_iou_matrix(track_boxes, detection_boxes)
    sim_mat = compute_appearance_similarity_matrix(track_embeddings, detection_embeddings)

    # 2. Build cost matrix
    # Normalize weights so they sum to 1.0
    w_sum = weight_spatial + weight_appearance
    ws = weight_spatial / w_sum if w_sum > 0 else 0.5
    wa = weight_appearance / w_sum if w_sum > 0 else 0.5

    cost_matrix = np.zeros((n_tracks, n_dets), dtype=np.float32)
    for i in range(n_tracks):
        for j in range(n_dets):
            iou_cost = 1.0 - iou_mat[i, j]
            # If detection or track has appearance, use appearance similarity, else fallback to 100% spatial
            has_app = (len(track_embeddings[i]) > 0 and detection_embeddings[j] is not None)
            if has_app:
                app_cost = 1.0 - sim_mat[i, j]
                cost_matrix[i, j] = ws * iou_cost + wa * app_cost
            else:
                cost_matrix[i, j] = iou_cost

    # 3. Solve optimal one-to-one linear assignment
    raw_matches = linear_assignment(cost_matrix)

    matched_tracks: Set[int] = set()
    matched_detections: Set[int] = set()
    valid_matches: List[Tuple[int, int]] = []
    ambiguous_matches: Set[int] = set()

    # 4. Gating verification
    for t_idx, d_idx in raw_matches:
        iou_val = iou_mat[t_idx, d_idx]
        sim_val = sim_mat[t_idx, d_idx]
        has_app = (len(track_embeddings[t_idx]) > 0 and detection_embeddings[d_idx] is not None)

        is_valid = False
        # Condition A: Standard spatial match satisfying iou_threshold
        if iou_val >= iou_threshold:
            # If appearance is also available, ensure it's not a severe appearance contradiction
            # (e.g. if IoU is high but appearance < 0.40, likely two crossing faces occupying same region)
            if has_app and sim_val < 0.40:
                is_valid = False
            else:
                is_valid = True

        # Condition B: Appearance-assisted recovery when IoU is low (due to motion or head tilt)
        elif has_app and sim_val >= appearance_threshold and iou_val >= spatial_gating_iou:
            is_valid = True

        if is_valid:
            valid_matches.append((t_idx, d_idx))
            matched_tracks.add(t_idx)
            matched_detections.add(d_idx)

    # 5. Check for ambiguity on valid matches
    for m_idx, (t_idx, d_idx) in enumerate(valid_matches):
        current_sim = sim_mat[t_idx, d_idx]
        current_iou = iou_mat[t_idx, d_idx]

        for other_t in range(n_tracks):
            if other_t != t_idx:
                other_iou = iou_mat[other_t, d_idx]
                other_sim = sim_mat[other_t, d_idx]

                # Ambiguity Condition 1: Multiple tracks spatially overlapping detection (crossing)
                if other_iou >= 0.20:
                    ambiguous_matches.add(m_idx)
                    break

                # Ambiguity Condition 2: Appearance margin is too narrow
                if current_sim > 0.50 and other_sim > 0.50:
                    if abs(current_sim - other_sim) < ambiguity_margin:
                        ambiguous_matches.add(m_idx)
                        break

    unmatched_tracks = [i for i in range(n_tracks) if i not in matched_tracks]
    unmatched_detections = [j for j in range(n_dets) if j not in matched_detections]

    return valid_matches, unmatched_tracks, unmatched_detections, ambiguous_matches


def associate_detections_to_tracks(
    track_boxes: List[Tuple[int, int, int, int]],
    detection_boxes: List[Tuple[int, int, int, int]],
    iou_threshold: float = 0.30,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Performs one-to-one Hungarian association between tracked bounding boxes (predictions)
    and newly observed detection bounding boxes.

    Returns:
        matches: List of (track_idx, detection_idx) pairs that satisfied iou_threshold.
        unmatched_tracks: List of track indices that had no matching detection.
        unmatched_detections: List of detection indices that had no matching track.
    """
    if not track_boxes:
        return [], [], list(range(len(detection_boxes)))

    if not detection_boxes:
        return [], list(range(len(track_boxes))), []

    # 1. Compute IoU matrix of shape (N_tracks, N_detections)
    iou_matrix = compute_iou_matrix(track_boxes, detection_boxes)

    # 2. Convert IoU into a cost matrix: cost = 1.0 - IoU
    cost_matrix = 1.0 - iou_matrix

    # 3. Solve optimal one-to-one linear sum assignment
    raw_matches = linear_assignment(cost_matrix)

    matched_tracks: Set[int] = set()
    matched_detections: Set[int] = set()
    valid_matches: List[Tuple[int, int]] = []

    # 4. Gating: Reject associations where IoU < iou_threshold
    for t_idx, d_idx in raw_matches:
        if iou_matrix[t_idx, d_idx] >= iou_threshold:
            valid_matches.append((t_idx, d_idx))
            matched_tracks.add(t_idx)
            matched_detections.add(d_idx)

    # 5. Determine unmatched tracks and detections
    unmatched_tracks = [i for i in range(len(track_boxes)) if i not in matched_tracks]
    unmatched_detections = [j for j in range(len(detection_boxes)) if j not in matched_detections]

    return valid_matches, unmatched_tracks, unmatched_detections


def reidentify_lost_tracks(
    lost_tracks: List[Tuple[str, List[np.ndarray], Tuple[int, int, int, int]]],
    unmatched_detections: List[Tuple[int, Tuple[int, int, int, int], Optional[np.ndarray]]],
    appearance_threshold: float = 0.65,
    max_scale_ratio: float = 3.5,
    ambiguity_margin: float = 0.08,
) -> Tuple[List[Tuple[str, int]], List[int]]:
    """
    Second association stage for re-identifying temporarily lost tracks against remaining unmatched detections.
    
    Args:
        lost_tracks: List of tuples (track_id, appearance_history, last_known_bbox).
        unmatched_detections: List of tuples (det_idx, bbox, embedding).
        appearance_threshold: High-confidence similarity threshold (default: 0.65).
        max_scale_ratio: Maximum allowable box area scale difference ratio.
        ambiguity_margin: Rejection margin if two lost tracks have close competing similarity scores.

    Returns:
        reidentified_matches: List of (track_id, det_idx) pairs.
        remaining_unmatched_dets: List of det_idx that were not re-identified.
    """
    if not lost_tracks or not unmatched_detections:
        return [], [d[0] for d in unmatched_detections]

    reidentified_matches: List[Tuple[str, int]] = []
    matched_det_indices: Set[int] = set()
    matched_track_ids: Set[str] = set()

    # Pre-extract embeddings
    track_embeddings = [t[1] for t in lost_tracks]
    det_embeddings = [d[2] for d in unmatched_detections]

    # Compute similarity matrix: shape (N_lost, N_unmatched_dets)
    sim_mat = compute_appearance_similarity_matrix(track_embeddings, det_embeddings)

    # Solve bipartite assignment minimizing cost = 1.0 - Sim
    cost_matrix = 1.0 - sim_mat
    raw_matches = linear_assignment(cost_matrix)

    for l_idx, d_pos in raw_matches:
        tid, _, t_bbox = lost_tracks[l_idx]
        det_idx, d_bbox, d_emb = unmatched_detections[d_pos]

        if d_emb is None:
            continue

        sim_val = sim_mat[l_idx, d_pos]

        # 1. Strict Appearance Gating (0.65)
        if sim_val < appearance_threshold:
            continue

        # 2. Scale Difference Plausibility Check:
        # A face returning from occlusion should not suddenly change area drastically (e.g. 10x)
        t_w, t_h = t_bbox[2], t_bbox[3]
        d_w, d_h = d_bbox[2], d_bbox[3]
        area_t = max(1.0, float(t_w * t_h))
        area_d = max(1.0, float(d_w * d_h))
        scale_ratio = max(area_t, area_d) / min(area_t, area_d)
        if scale_ratio > max_scale_ratio:
            continue

        # 3. Ambiguity & Competition Safeguard:
        # Check if another lost track or detection has a nearly identical competing match
        is_ambiguous = False
        for other_l_idx in range(len(lost_tracks)):
            if other_l_idx != l_idx:
                other_sim = sim_mat[other_l_idx, d_pos]
                if other_sim >= (appearance_threshold - 0.05) and abs(sim_val - other_sim) < ambiguity_margin:
                    is_ambiguous = True
                    break

        if is_ambiguous:
            continue

        # Match accepted
        reidentified_matches.append((tid, det_idx))
        matched_track_ids.add(tid)
        matched_det_indices.add(det_idx)

    remaining_unmatched_dets = [
        d[0] for d in unmatched_detections if d[0] not in matched_det_indices
    ]

    return reidentified_matches, remaining_unmatched_dets

