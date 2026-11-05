"""
ReliAttend AI — Multi-Face Tracking Service (Phase 9 - Substage 9.3)
Manages temporary in-memory tracking sessions, coordinates frame ingestion,
runs YuNet multi-face detection, feeds detections to MultiFaceTracker,
and enforces authorization, ownership, session timeouts, and sequence monotonicity.
Strictly safeguards privacy: no frames or biometric embeddings are stored or exposed.
"""

from dataclasses import dataclass, field
import logging
import os
import secrets
import threading
import time
from typing import Dict, List, Optional, Set, Tuple
import cv2
from fastapi import UploadFile
import numpy as np
from sqlalchemy.orm import Session

from app.ai.detector import FaceDetector
from app.ai.matcher import CandidateMatch, FaceMatchResult, GalleryItem, VectorMatcher
from app.ai.recognition import FaceRecognizer
from app.ai.tracking.track import RecognitionObservation, TrackState
from app.ai.tracking.tracker import DetectionInput, MultiFaceTracker, TrackOutput
from app.config import get_settings
from app.core.exceptions import (
    AuthorizationForbiddenError,
    BadRequestError,
    DuplicateResourceError,
    ResourceNotFoundError,
)
from app.models.academic import ClassSection, FacultyAssignment
from app.models.student import ConsentRecord, FaceReference, Student
from app.models.user import User
from app.schemas.detection import BoundingBoxSchema
from app.schemas.tracking import (
    CreateTrackingSessionRequest,
    EndTrackingSessionResponse,
    ProcessFrameTrackingResponse,
    RecognizedStudentMetadata,
    SessionTracksListResponse,
    TrackItemResponse,
    TrackingConfigSchema,
    TrackingSessionResponse,
)

logger = logging.getLogger("reliattend.tracking")
settings = get_settings()


@dataclass
class TrackingSession:
    """In-memory representation of an active face-tracking session."""
    session_id: str
    user_id: int
    user_role: str
    class_section_id: int
    tracker: MultiFaceTracker
    created_at: float
    last_activity_at: float
    gallery: List[GalleryItem] = field(default_factory=list)
    matcher: Optional[VectorMatcher] = None
    gallery_generation: int = 1
    recognition_enabled: bool = True
    lock: threading.Lock = field(default_factory=threading.Lock)


class TrackingSessionManager:
    """
    Thread-safe in-memory session registry for multi-face tracking.
    Enforces maximum idle timeout and concurrent session limits.
    """

    def __init__(self, idle_timeout_seconds: float = 300.0, max_active_sessions: int = 50):
        self._sessions: Dict[str, TrackingSession] = {}
        self._lock = threading.Lock()
        self.idle_timeout_seconds = idle_timeout_seconds
        self.max_active_sessions = max_active_sessions

    def create_session(
        self,
        user_id: int,
        user_role: str,
        class_section_id: int,
        config: CreateTrackingSessionRequest,
        gallery: Optional[List[GalleryItem]] = None,
    ) -> TrackingSession:
        with self._lock:
            self._cleanup_expired_locked()

            if len(self._sessions) >= self.max_active_sessions:
                raise DuplicateResourceError("Maximum concurrent tracking sessions reached. Please close inactive sessions.")

            # Generate random session ID
            session_id = f"trk_sess_{secrets.token_hex(8)}"
            now = time.time()

            tracker = MultiFaceTracker(
                iou_threshold=config.iou_threshold or 0.30,
                min_hits_to_confirm=config.min_hits_to_confirm or 3,
                max_missing_frames=config.max_missing_frames or 15,
                max_missing_seconds=config.max_missing_seconds or 3.0,
                ambiguity_iou_threshold=config.ambiguity_iou_threshold or 0.20,
                appearance_threshold=config.appearance_threshold or 0.60,
                weight_spatial=config.weight_spatial if config.weight_spatial is not None else 0.50,
                weight_appearance=config.weight_appearance if config.weight_appearance is not None else 0.50,
            )

            sess_gallery = gallery or []
            matcher = VectorMatcher(gallery=sess_gallery, similarity_threshold=0.65, ambiguity_margin=0.05)

            session = TrackingSession(
                session_id=session_id,
                user_id=user_id,
                user_role=user_role,
                class_section_id=class_section_id,
                tracker=tracker,
                created_at=now,
                last_activity_at=now,
                gallery=sess_gallery,
                matcher=matcher,
            )
            self._sessions[session_id] = session
            logger.info("Created tracking session %s for user %d (section %d, %d gallery refs)", session_id, user_id, class_section_id, len(sess_gallery))
            return session

    def get_session(self, session_id: str) -> Optional[TrackingSession]:
        with self._lock:
            self._cleanup_expired_locked()
            return self._sessions.get(session_id)

    def remove_session(self, session_id: str) -> Optional[TrackingSession]:
        with self._lock:
            return self._sessions.pop(session_id, None)

    def invalidate_student_references(
        self,
        student_id: Optional[int] = None,
        student_str_id: Optional[str] = None,
        class_section_id: Optional[int] = None,
        reference_id: Optional[int] = None,
    ) -> int:
        """
        Thread-safely removes reference embeddings and clears accumulated track identities
        for affected students across all active tracking sessions.

        Returns total count of sessions updated.
        """
        affected_count = 0
        with self._lock:
            sessions = list(self._sessions.values())

        for session in sessions:
            if class_section_id is not None and session.class_section_id != class_section_id:
                continue

            with session.lock:
                removed_student_ids: Set[str] = set()
                new_gallery: List[GalleryItem] = []

                for item in session.gallery:
                    should_remove = False
                    if reference_id is not None and item.reference_id == reference_id:
                        should_remove = True
                    elif student_id is not None and item.student_id == student_id:
                        should_remove = True
                    elif student_str_id is not None and item.student_str_id == student_str_id:
                        should_remove = True

                    if should_remove:
                        removed_student_ids.add(item.student_str_id)
                    else:
                        new_gallery.append(item)

                if removed_student_ids or len(new_gallery) != len(session.gallery):
                    session.gallery = new_gallery
                    session.matcher = VectorMatcher(
                        gallery=new_gallery,
                        similarity_threshold=0.65,
                        ambiguity_margin=0.05,
                    )
                    session.gallery_generation += 1
                    # Clear recognition accumulators on all active tracks holding removed identities
                    for trk in session.tracker.tracks.values():
                        ident = trk.identity_accumulator
                        if (
                            ident.assigned_student_id in removed_student_ids
                            or any(obs and obs.student_str_id in removed_student_ids for obs in ident.history)
                        ):
                            ident.clear()

                    affected_count += 1
                    logger.info(
                        "Invalidated %d student reference(s) in session %s (remaining gallery: %d, new gen: %d)",
                        len(removed_student_ids),
                        session.session_id,
                        len(new_gallery),
                        session.gallery_generation,
                    )

        return affected_count

    def terminate_sessions_for_user(self, user_id: int) -> int:
        """Terminates all active sessions owned by user_id (e.g. on faculty permission revocation)."""
        terminated = 0
        with self._lock:
            sids = [sid for sid, s in self._sessions.items() if s.user_id == user_id]
            for sid in sids:
                sess = self._sessions.pop(sid, None)
                if sess:
                    with sess.lock:
                        sess.tracker.terminate_all()
                        sess.gallery.clear()
                        sess.matcher = None
                    terminated += 1
        return terminated

    def terminate_sessions_for_section(self, class_section_id: int) -> int:
        """Terminates all active sessions for a class section."""
        terminated = 0
        with self._lock:
            sids = [sid for sid, s in self._sessions.items() if s.class_section_id == class_section_id]
            for sid in sids:
                sess = self._sessions.pop(sid, None)
                if sess:
                    with sess.lock:
                        sess.tracker.terminate_all()
                        sess.gallery.clear()
                        sess.matcher = None
                    terminated += 1
        return terminated

    def mark_sessions_unavailable_for_section(self, class_section_id: int) -> int:
        """
        Fail-safe fallback: disables automatic recognition and purges galleries for active sessions
        in a section if an unexpected error occurred during granular invalidation.
        """
        count = 0
        with self._lock:
            sessions = [s for s in self._sessions.values() if s.class_section_id == class_section_id]

        for s in sessions:
            with s.lock:
                s.recognition_enabled = False
                s.gallery.clear()
                s.matcher = None
                s.gallery_generation += 1
                for trk in s.tracker.tracks.values():
                    trk.identity_accumulator.clear()
                count += 1
        return count

    def _cleanup_expired_locked(self) -> None:
        """Removes sessions that have been idle longer than idle_timeout_seconds."""
        now = time.time()
        expired_ids = [
            sid for sid, s in self._sessions.items()
            if (now - s.last_activity_at) > self.idle_timeout_seconds
        ]
        for sid in expired_ids:
            logger.info("Purging expired tracking session %s (idle > %ds)", sid, self.idle_timeout_seconds)
            self._sessions.pop(sid, None)

    def clear_all(self) -> None:
        """Utility for test suites to clear all active sessions."""
        with self._lock:
            self._sessions.clear()


# Global in-memory session manager singleton
session_manager = TrackingSessionManager()


class TrackingService:
    """
    Coordinates multi-face tracking sessions, frame detection, and state tracking.
    """

    def __init__(
        self,
        db: Session,
        detector: Optional[FaceDetector] = None,
        recognizer: Optional[FaceRecognizer] = None,
        manager: Optional[TrackingSessionManager] = None,
    ):
        self.db = db
        self.manager = manager or session_manager

        if detector is not None:
            self.detector = detector
        else:
            yunet_path = os.path.join(
                settings.BIOMETRIC_MODELS_DIR, "face_detection_yunet_2023mar.onnx"
            )
            self.detector = FaceDetector(
                model_path=yunet_path,
                confidence_threshold=settings.DETECTION_DEFAULT_CONFIDENCE_THRESHOLD,
                nms_threshold=settings.DETECTION_DEFAULT_NMS_THRESHOLD,
                min_face_width=settings.BIOMETRIC_MIN_FACE_WIDTH,
                min_face_height=settings.BIOMETRIC_MIN_FACE_HEIGHT,
                blur_threshold=settings.BIOMETRIC_BLUR_THRESHOLD,
                min_brightness=settings.BIOMETRIC_MIN_BRIGHTNESS,
                max_brightness=settings.BIOMETRIC_MAX_BRIGHTNESS,
            )

        if recognizer is not None:
            self.recognizer = recognizer
        else:
            sface_path = os.path.join(
                settings.BIOMETRIC_MODELS_DIR, "face_recognition_sface_2021dec.onnx"
            )
            self.recognizer = FaceRecognizer(model_path=sface_path)

    def _validate_faculty_access(self, class_section_id: int, current_user: User) -> ClassSection:
        """Ensures the class section exists and current faculty is assigned to it."""
        section = (
            self.db.query(ClassSection)
            .filter(ClassSection.id == class_section_id, ClassSection.is_active == True)
            .first()
        )
        if not section:
            raise ResourceNotFoundError(f"Class section {class_section_id} not found or inactive")

        if current_user.role != "admin":
            assignment = (
                self.db.query(FacultyAssignment)
                .filter(
                    FacultyAssignment.user_id == current_user.id,
                    FacultyAssignment.class_section_id == class_section_id,
                    FacultyAssignment.is_active == True,
                )
                .first()
            )
            if not assignment:
                raise AuthorizationForbiddenError(
                    f"Faculty member {current_user.email} is not authorized for class section {class_section_id}"
                )

        return section

    def _get_and_validate_session(self, session_id: str, current_user: User) -> TrackingSession:
        """Retrieves session and verifies requester ownership/permissions."""
        session = self.manager.get_session(session_id)
        if not session:
            raise ResourceNotFoundError(f"Tracking session '{session_id}' not found or has expired")

        # Ownership check: session creator or Admin
        if current_user.role != "admin" and session.user_id != current_user.id:
            raise AuthorizationForbiddenError("You are not authorized to access or modify this tracking session")

        return session

    def _load_section_gallery(self, class_section_id: int) -> List[GalleryItem]:
        """
        Loads all active reference embeddings for students enrolled in the class section.
        Strict DPDP rule: Excludes students without active consent.
        """
        if self.db is None:
            return []

        students = (
            self.db.query(Student)
            .filter_by(class_section_id=class_section_id, is_active=True)
            .all()
        )

        gallery: List[GalleryItem] = []
        for student in students:
            # Check consent
            active_consent = (
                self.db.query(ConsentRecord)
                .filter_by(student_id=student.id, is_active=True)
                .first()
            )
            if not active_consent:
                continue

            active_refs = (
                self.db.query(FaceReference)
                .filter_by(student_id=student.id, status="active")
                .all()
            )

            for ref in active_refs:
                try:
                    emb = FaceRecognizer.deserialize_embedding(ref.embedding)
                    gallery.append(
                        GalleryItem(
                            reference_id=ref.id,
                            student_id=student.id,
                            student_str_id=student.student_id,
                            student_name=student.name,
                            pose_label=ref.pose_label or "frontal",
                            is_primary=ref.is_primary,
                            embedding=emb,
                        )
                    )
                except Exception as e:
                    logger.warning("Skipping corrupted embedding for reference_id=%d: %s", ref.id, str(e))

        return gallery

    def create_session(
        self,
        request: CreateTrackingSessionRequest,
        current_user: User,
    ) -> TrackingSessionResponse:
        """Creates a new tracking session after validating academic authorization."""
        self._validate_faculty_access(request.class_section_id, current_user)
        gallery = self._load_section_gallery(request.class_section_id)

        session = self.manager.create_session(
            user_id=current_user.id,
            user_role=current_user.role,
            class_section_id=request.class_section_id,
            config=request,
            gallery=gallery,
        )

        cfg = TrackingConfigSchema(
            iou_threshold=session.tracker.iou_threshold,
            min_hits_to_confirm=session.tracker.min_hits_to_confirm,
            max_missing_frames=session.tracker.max_missing_frames,
            max_missing_seconds=session.tracker.max_missing_seconds,
            ambiguity_iou_threshold=session.tracker.ambiguity_iou_threshold,
            appearance_threshold=session.tracker.appearance_threshold,
            weight_spatial=session.tracker.weight_spatial,
            weight_appearance=session.tracker.weight_appearance,
        )

        return TrackingSessionResponse(
            session_id=session.session_id,
            class_section_id=session.class_section_id,
            created_at=session.created_at,
            config=cfg,
        )

    async def process_frame(
        self,
        session_id: str,
        frame_seq: int,
        timestamp: Optional[float],
        file: UploadFile,
        current_user: User,
    ) -> ProcessFrameTrackingResponse:
        """
        Processes a single sequential image frame in the session:
        1. Validates ownership and session state.
        2. Safely decodes frame in memory.
        3. Detects faces via YuNet.
        4. Updates MultiFaceTracker.
        5. Returns updated tracking state without persisting frame bytes.
        """
        session = self._get_and_validate_session(session_id, current_user)

        if frame_seq < 0:
            raise BadRequestError("frame_seq must be a non-negative integer")

        frame_timestamp = timestamp if timestamp is not None else time.time()

        # Enforce frame sequence monotonically increasing
        if (
            session.tracker.last_processed_frame_seq is not None
            and frame_seq <= session.tracker.last_processed_frame_seq
        ):
            raise BadRequestError(
                f"frame_seq ({frame_seq}) must be strictly greater than last processed frame "
                f"({session.tracker.last_processed_frame_seq})"
            )

        # 1. Validate MIME type
        if not file.content_type or file.content_type not in settings.BIOMETRIC_ALLOWED_MIME_TYPES:
            raise BadRequestError(
                f"Unsupported or missing content type '{file.content_type}'. "
                f"Allowed types: {', '.join(settings.BIOMETRIC_ALLOWED_MIME_TYPES)}"
            )

        # 2. Read bytes safely with size check
        image_bytes = await file.read()
        if not image_bytes:
            raise BadRequestError("Uploaded frame file is empty")

        if len(image_bytes) > settings.BIOMETRIC_MAX_UPLOAD_BYTES:
            raise BadRequestError(
                f"Frame image size ({len(image_bytes)} bytes) exceeds maximum limit "
                f"({settings.BIOMETRIC_MAX_UPLOAD_BYTES} bytes)"
            )

        # 3. Decode image completely in-memory
        try:
            image_np = self.detector.decode_image_bytes(image_bytes)
            if image_np is None:
                raise BadRequestError("Failed to decode frame image: invalid or unsupported image format")
        except BadRequestError:
            raise
        except Exception as e:
            raise BadRequestError(f"Failed to decode frame image: {str(e)}")

        # 4. Validate decoded frame dimensions
        orig_h, orig_w = image_np.shape[:2]
        if orig_w > settings.DETECTION_MAX_FRAME_WIDTH or orig_h > settings.DETECTION_MAX_FRAME_HEIGHT:
            raise BadRequestError(
                f"Frame dimensions ({orig_w}x{orig_h}) exceed maximum permitted size "
                f"({settings.DETECTION_MAX_FRAME_WIDTH}x{settings.DETECTION_MAX_FRAME_HEIGHT})"
            )

        # Detect faces in frame
        det_result = self.detector.detect_multi_faces(image_np)

        # Convert detected faces to DetectionInput and extract appearance embedding (in-memory only)
        detection_inputs: List[DetectionInput] = []
        for face in det_result.faces:
            emb_vector: Optional[np.ndarray] = None
            if face.is_quality_acceptable and face.face_data is not None:
                try:
                    emb_res = self.recognizer.extract_embedding(image_np, face.face_data)
                    if emb_res.is_valid:
                        emb_vector = FaceRecognizer.deserialize_embedding(emb_res.embedding_bytes)
                except Exception as exc:
                    logger.debug("SFace extraction skipped on tracking face_id=%d: %s", face.face_id, exc)

            detection_inputs.append(
                DetectionInput(
                    face_id=face.face_id,
                    bounding_box=face.bounding_box,
                    confidence=face.confidence,
                    landmarks=face.landmarks,
                    embedding=emb_vector,
                    is_quality_acceptable=face.is_quality_acceptable,
                )
            )

        # 1:N Recognition Matching against session cached gallery (single-pass, zero redundant SFace)
        # Capture atomic snapshot of gallery matcher and generation under lock
        recog_results_by_face_id: Dict[int, FaceMatchResult] = {}
        matched_generation: Optional[int] = None
        current_matcher: Optional[VectorMatcher] = None

        with session.lock:
            if session.recognition_enabled and session.matcher and session.gallery:
                matched_generation = session.gallery_generation
                current_matcher = session.matcher

        if current_matcher is not None:
            live_embs: List[Tuple[int, np.ndarray]] = [
                (det.face_id, det.embedding)
                for det in detection_inputs
                if det.embedding is not None and det.is_quality_acceptable
            ]
            if live_embs:
                recog_list = current_matcher.match_embeddings(live_embs)
                for r in recog_list:
                    recog_results_by_face_id[r.face_id] = r

        # Thread-safe tracker update and identity resolution
        with session.lock:
            # Race condition verification: if gallery generation advanced or recognition was disabled
            # while embeddings were being matched, discard stale observations
            is_stale_recognition = (
                not session.recognition_enabled
                or (matched_generation is not None and matched_generation != session.gallery_generation)
            )
            if is_stale_recognition and recog_results_by_face_id:
                logger.info(
                    "Discarding stale in-flight recognition results for session %s (matched gen %s vs current %s)",
                    session.session_id,
                    matched_generation,
                    session.gallery_generation,
                )
                recog_results_by_face_id.clear()

            tracking_result = session.tracker.update(
                detections=detection_inputs,
                frame_seq=frame_seq,
                timestamp=frame_timestamp,
            )

            # Update Track Recognition Identity Accumulators
            # Map detections to updated tracks using explicit matched_face_id
            active_assigned_students: Dict[str, str] = {}  # student_str_id -> track_id
            for t_out in tracking_result.tracks:
                trk = session.tracker.tracks.get(t_out.track_id)
                if not trk or trk.state == TrackState.TERMINATED:
                    continue

                # Locate recognition observation via explicit matched_face_id assigned by tracker
                obs: Optional[RecognitionObservation] = None
                if t_out.matched_face_id is not None and t_out.matched_face_id in recog_results_by_face_id:
                    m_res = recog_results_by_face_id[t_out.matched_face_id]
                    if m_res.match_status == "matched" and m_res.top_match is not None:
                        obs = RecognitionObservation(
                            student_str_id=m_res.top_match.student_str_id,
                            student_name=m_res.top_match.student_name,
                            similarity=m_res.top_match.similarity_score,
                            frame_seq=frame_seq,
                            timestamp=frame_timestamp,
                        )

                # Feed observation to track identity accumulator (pauses on ambiguity)
                trk.identity_accumulator.add_observation(
                    observation=obs,
                    track_state=trk.state,
                    is_ambiguous=t_out.is_ambiguous,
                )

                # Enforce one active track per assigned student (conflict prevention)
                ident = trk.identity_accumulator
                if ident.identity_status == "assigned" and ident.assigned_student_id:
                    if ident.assigned_student_id in active_assigned_students and active_assigned_students[ident.assigned_student_id] != trk.track_id:
                        ident.mark_conflict()
                    else:
                        active_assigned_students[ident.assigned_student_id] = trk.track_id

            session.last_activity_at = time.time()

        # Map tracker outputs to response schema
        track_items: List[TrackItemResponse] = []
        for t in tracking_result.tracks:
            x, y, w, h = t.bounding_box
            trk = session.tracker.tracks.get(t.track_id)
            ident = trk.identity_accumulator if trk else None

            recog_meta: Optional[RecognizedStudentMetadata] = None
            masked_id: Optional[str] = None

            if ident and ident.assigned_student_id:
                # Mask student ID: 'STU2026001' -> 'STU...001'
                sid = ident.assigned_student_id
                if len(sid) > 6:
                    masked_id = f"{sid[:3]}...{sid[-3:]}"
                else:
                    masked_id = sid

                recog_meta = RecognizedStudentMetadata(
                    student_id_masked=masked_id,
                    student_name=ident.assigned_student_name,
                    recognition_similarity=ident.recognition_similarity or 0.0,
                    identity_status=ident.identity_status,
                    consistent_hits=len([o for o in ident.history if o and o.student_str_id == ident.assigned_student_id]),
                )
            elif ident and ident.identity_status == "conflict":
                recog_meta = RecognizedStudentMetadata(
                    student_id_masked="CONFLICT",
                    student_name=None,
                    recognition_similarity=ident.recognition_similarity or 0.0,
                    identity_status="conflict",
                    consistent_hits=0,
                )

            track_items.append(
                TrackItemResponse(
                    track_id=t.track_id,
                    state=t.state,
                    bounding_box=BoundingBoxSchema(x=x, y=y, width=w, height=h),
                    detection_confidence=round(t.detection_confidence, 4),
                    total_observations=t.total_observations,
                    consecutive_misses=t.consecutive_misses,
                    first_seen_frame=t.first_seen_frame,
                    last_seen_frame=t.last_seen_frame,
                    first_seen_timestamp=round(t.first_seen_timestamp, 4),
                    last_seen_timestamp=round(t.last_seen_timestamp, 4),
                    is_newly_created=t.is_newly_created,
                    is_recovered=t.is_recovered,
                    is_ambiguous=t.is_ambiguous,
                    associated_student_id=masked_id,
                    recognized_student=recog_meta,
                )
            )

        return ProcessFrameTrackingResponse(
            session_id=session.session_id,
            frame_seq=tracking_result.frame_seq,
            timestamp=round(tracking_result.timestamp, 4),
            active_tracks_count=tracking_result.active_tracks_count,
            confirmed_tracks_count=tracking_result.confirmed_tracks_count,
            unmatched_detections_count=tracking_result.unmatched_detections_count,
            tracks=track_items,
        )

    def get_tracks(self, session_id: str, current_user: User) -> SessionTracksListResponse:
        """Returns all currently active (non-terminated) tracks in the session."""
        session = self._get_and_validate_session(session_id, current_user)

        with session.lock:
            active_tracks = session.tracker.get_active_tracks()
            session.last_activity_at = time.time()

        track_items: List[TrackItemResponse] = []
        for t in active_tracks:
            x, y, w, h = t.bounding_box
            trk = session.tracker.tracks.get(t.track_id)
            ident = trk.identity_accumulator if trk else None

            recog_meta: Optional[RecognizedStudentMetadata] = None
            masked_id: Optional[str] = None

            if ident and ident.assigned_student_id:
                sid = ident.assigned_student_id
                masked_id = f"{sid[:3]}...{sid[-3:]}" if len(sid) > 6 else sid
                recog_meta = RecognizedStudentMetadata(
                    student_id_masked=masked_id,
                    student_name=ident.assigned_student_name,
                    recognition_similarity=ident.recognition_similarity or 0.0,
                    identity_status=ident.identity_status,
                    consistent_hits=len([o for o in ident.history if o and o.student_str_id == ident.assigned_student_id]),
                )
            elif ident and ident.identity_status == "conflict":
                recog_meta = RecognizedStudentMetadata(
                    student_id_masked="CONFLICT",
                    student_name=None,
                    recognition_similarity=ident.recognition_similarity or 0.0,
                    identity_status="conflict",
                    consistent_hits=0,
                )

            track_items.append(
                TrackItemResponse(
                    track_id=t.track_id,
                    state=t.state,
                    bounding_box=BoundingBoxSchema(x=x, y=y, width=w, height=h),
                    detection_confidence=round(t.detection_confidence, 4),
                    total_observations=t.total_observations,
                    consecutive_misses=t.consecutive_misses,
                    first_seen_frame=t.first_seen_frame,
                    last_seen_frame=t.last_seen_frame,
                    first_seen_timestamp=round(t.first_seen_timestamp, 4),
                    last_seen_timestamp=round(t.last_seen_timestamp, 4),
                    is_newly_created=t.is_newly_created,
                    is_recovered=t.is_recovered,
                    is_ambiguous=t.is_ambiguous,
                    associated_student_id=masked_id,
                    recognized_student=recog_meta,
                )
            )

        return SessionTracksListResponse(
            session_id=session.session_id,
            active_tracks_count=len(track_items),
            tracks=track_items,
        )

    def end_session(self, session_id: str, current_user: User) -> EndTrackingSessionResponse:
        """Terminates tracking session and frees memory resources."""
        session = self._get_and_validate_session(session_id, current_user)

        with session.lock:
            # Terminate all tracks in tracker
            session.tracker.terminate_all()
            session.gallery.clear()
            session.matcher = None

        total_created = session.tracker.next_track_number - 1
        duration = time.time() - session.created_at

        # Remove from active sessions
        self.manager.remove_session(session_id)
        logger.info("Ended tracking session %s. Total tracks created: %d", session_id, total_created)

        return EndTrackingSessionResponse(
            session_id=session_id,
            message=f"Tracking session {session_id} successfully terminated",
            total_tracks_created=total_created,
            duration_seconds=round(duration, 2),
        )
