"""
ReliAttend AI — Vectorized Biometric Feature Matcher (Phase 8)
Computes cosine similarity between live face embeddings and enrolled student gallery.
Implements multi-reference aggregation (max-pooling), top-K ranking, ambiguity margin
calculation, and duplicate identity conflict detection.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class GalleryItem:
    """Represents a single active reference embedding in the gallery."""
    reference_id: int
    student_id: int  # Database PK
    student_str_id: str  # Institutional string ID e.g. 'STU2026001'
    student_name: str
    pose_label: str
    is_primary: bool
    embedding: np.ndarray  # (128,) float32, L2-normalized


@dataclass
class CandidateMatch:
    """Match result for an individual student candidate."""
    student_id: int
    student_str_id: str
    student_name: str
    similarity_score: float
    matched_pose: str
    is_primary: bool


@dataclass
class FaceMatchResult:
    """Matching classification for a single live face embedding."""
    face_id: int
    match_status: str  # "matched", "uncertain", "unrecognized"
    top_match: Optional[CandidateMatch]
    top_candidates: List[CandidateMatch]
    margin_to_second: Optional[float] = None
    reason: Optional[str] = None


class VectorMatcher:
    """
    High-performance vectorized 1:N face embedding matcher.
    Uses BLAS matrix multiplication for O(1) similarity matrix calculation.
    """

    def __init__(
        self,
        gallery: List[GalleryItem],
        similarity_threshold: float = 0.65,
        ambiguity_margin: float = 0.05,
        top_k: int = 3,
    ):
        self.gallery = gallery
        self.similarity_threshold = similarity_threshold
        self.ambiguity_margin = ambiguity_margin
        self.top_k = top_k

        # Build gallery matrix (N_refs, 128)
        if gallery:
            self.gallery_matrix = np.vstack([item.embedding for item in gallery]).astype(np.float32)
        else:
            self.gallery_matrix = np.empty((0, 128), dtype=np.float32)

    def match_embeddings(
        self,
        live_embeddings: List[Tuple[int, np.ndarray]],  # List of (face_id, embedding_vector)
    ) -> List[FaceMatchResult]:
        """
        Compare live embeddings against gallery and apply matching rules.
        Handles empty galleries, small embeddings, ambiguity, and duplicate identities.
        """
        results: List[FaceMatchResult] = []

        if not live_embeddings:
            return results

        if len(self.gallery) == 0 or self.gallery_matrix.shape[0] == 0:
            for face_id, _ in live_embeddings:
                results.append(
                    FaceMatchResult(
                        face_id=face_id,
                        match_status="unrecognized",
                        top_match=None,
                        top_candidates=[],
                        margin_to_second=None,
                        reason="No active reference embeddings enrolled in section gallery.",
                    )
                )
            return results

        # 1. Stack live embeddings into matrix (N_live, 128)
        live_matrix = np.vstack([emb for _, emb in live_embeddings]).astype(np.float32)

        # 2. Compute similarity matrix (N_live, N_gallery) via dot product
        # Both live and gallery embeddings are L2-normalized so dot product == cosine similarity
        sim_matrix = np.dot(live_matrix, self.gallery_matrix.T)

        # 3. Process each live face
        for live_idx, (face_id, _) in enumerate(live_embeddings):
            row_sims = sim_matrix[live_idx]  # Similarities across all gallery references

            # Aggregate per student: compute max similarity across their poses
            student_scores: Dict[int, Dict] = {}
            for g_idx, sim in enumerate(row_sims):
                sim_float = float(sim)
                g_item = self.gallery[g_idx]
                s_id = g_item.student_id

                if s_id not in student_scores or sim_float > student_scores[s_id]["score"]:
                    student_scores[s_id] = {
                        "student_id": g_item.student_id,
                        "student_str_id": g_item.student_str_id,
                        "student_name": g_item.student_name,
                        "score": sim_float,
                        "matched_pose": g_item.pose_label,
                        "is_primary": g_item.is_primary,
                    }

            # Sort students by similarity descending
            sorted_students = sorted(
                student_scores.values(),
                key=lambda item: item["score"],
                reverse=True,
            )

            # Build top candidates
            top_candidates = [
                CandidateMatch(
                    student_id=c["student_id"],
                    student_str_id=c["student_str_id"],
                    student_name=c["student_name"],
                    similarity_score=round(c["score"], 4),
                    matched_pose=c["matched_pose"],
                    is_primary=c["is_primary"],
                )
                for c in sorted_students[: self.top_k]
            ]

            if not top_candidates:
                results.append(
                    FaceMatchResult(
                        face_id=face_id,
                        match_status="unrecognized",
                        top_match=None,
                        top_candidates=[],
                        margin_to_second=None,
                        reason="No valid candidate scores evaluated.",
                    )
                )
                continue

            top_1 = top_candidates[0]
            # Calculate margin between highest-scoring distinct students
            # Requirement 1 & 2: Return null only when fewer than two distinct student candidates exist.
            # When >= 2 distinct students exist, compute margin = top_1 - top_2 regardless of whether face is matched or unrecognized.
            if len(top_candidates) > 1:
                top_2_score = top_candidates[1].similarity_score
                margin = round(top_1.similarity_score - top_2_score, 4)
            else:
                top_2_score = None
                margin = None

            # Evaluate decision rules
            if top_1.similarity_score < self.similarity_threshold:
                match_status = "unrecognized"
                top_match = None
                reason = f"Top similarity score ({top_1.similarity_score}) below threshold ({self.similarity_threshold})."
            elif margin is not None and margin < self.ambiguity_margin:
                # Ambiguous collision: top 2 distinct student candidates are too close in score
                match_status = "uncertain"
                top_match = top_1
                reason = (
                    f"Ambiguous match: top score ({top_1.similarity_score}) and second score "
                    f"({top_2_score}) differ by {margin}, which is below ambiguity margin ({self.ambiguity_margin})."
                )
            else:
                match_status = "matched"
                top_match = top_1
                reason = None

            results.append(
                FaceMatchResult(
                    face_id=face_id,
                    match_status=match_status,
                    top_match=top_match,
                    top_candidates=top_candidates,
                    margin_to_second=margin,
                    reason=reason,
                )
            )

        # 4. Resolve duplicate identity collisions across the frame
        # Requirement 3: If multiple detected faces match the same student, flag as uncertain
        self._resolve_duplicate_identities(results)

        return results

    def _resolve_duplicate_identities(self, results: List[FaceMatchResult]) -> None:
        """
        Flags multiple detections matching the same student identity as uncertain.
        In a single frame, a real human can only occupy one face bounding box.
        """
        student_occurrences: Dict[int, List[FaceMatchResult]] = {}
        for r in results:
            if r.top_match and r.match_status == "matched":
                sid = r.top_match.student_id
                student_occurrences.setdefault(sid, []).append(r)

        for sid, matched_list in student_occurrences.items():
            if len(matched_list) > 1:
                # Conflict detected: 2 or more faces claimed the same student
                s_name = matched_list[0].top_match.student_name if matched_list[0].top_match else str(sid)
                for r in matched_list:
                    r.match_status = "uncertain"
                    r.reason = (
                        f"Duplicate identity conflict: multiple faces in frame matched student "
                        f"'{s_name}' (ID: {sid}). Flagged as uncertain."
                    )

