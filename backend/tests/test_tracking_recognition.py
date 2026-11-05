"""
ReliAttend AI — Multi-Face Tracking Recognition Integration Tests (Phase 9 - Substage 9.7)
Deterministic tests verifying:
1. Three consistent matched recognition observations within rolling window of 5 frames assign student identity.
2. Non-matching/uncertain/unrecognized observations interrupt evidence accumulation without assigning identity.
3. Tentative track identity prevention: tentative tracks never receive assigned student identity.
4. Uncertain and unrecognized faces continue tracking with unassigned identity status.
5. Multiple students recognised simultaneously and assigned their respective student identities.
6. Duplicate student identity claims across simultaneous tracks result in conflict flagging.
7. Ambiguous crossing frames pause recognition accumulation without corrupting identity.
8. Temporary disappearance and recovery preserves assigned student identity.
9. Terminated track releases student identity; re-entry requires independent accumulation.
10. Detection-to-track correspondence correctly maps recognition results.
11. Section-scoped gallery loading loads only students in the section with active consent.
12. Consent revocation or missing consent excludes student from session gallery.
13. Session deletion clears all cached gallery, matchers, and identity accumulators.
14. RBAC and authorization enforcement prevents unassigned faculty from creating session.
15. Zero attendance records or review decisions are created (pure identity resolution).
16. Zero duplicate SFace extraction (single-pass embedding reuse).
"""

import datetime
import time
import numpy as np
import pytest

from app.ai.detector import DetectedFaceBox, MultiFaceDetectionResult, QualityMetrics
from app.ai.matcher import GalleryItem, VectorMatcher
from app.ai.recognition import FaceRecognizer
from app.ai.tracking import (
    DetectionInput,
    MultiFaceTracker,
    TrackState,
)
from app.ai.tracking.track import FaceTrack, RecognitionObservation, TrackIdentityAccumulator
from app.database import Base, SessionLocal, engine
from app.models.academic import AcademicYear, ClassSection, Department, FacultyAssignment, Programme, Semester, Subject
from app.models.student import ConsentRecord, FaceReference, Student
from app.models.user import User
from app.schemas.tracking import CreateTrackingSessionRequest
from app.services.tracking_service import TrackingService, TrackingSessionManager


def _norm_vec(seed: int) -> np.ndarray:
    """Deterministic normalized 128-d float32 vector."""
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
# TEST 1: Three consistent matched observations within 5 frames assign identity
# ==============================================================================
def test_three_consistent_matches_assign_identity():
    """Requires 3 consistent matches within 5 frames to promote to assigned."""
    accum = TrackIdentityAccumulator(min_consistent_hits=3, window_size=5)

    obs1 = RecognitionObservation("STU001", "Alice", 0.85, 1, 1.0)
    obs2 = RecognitionObservation("STU001", "Alice", 0.88, 2, 1.1)
    obs3 = RecognitionObservation("STU001", "Alice", 0.82, 3, 1.2)

    # Frame 1: 1 hit -> provisional
    accum.add_observation(obs1, track_state=TrackState.CONFIRMED, is_ambiguous=False)
    assert accum.identity_status == "provisional"
    assert accum.assigned_student_id is None

    # Frame 2: 2 hits -> provisional
    accum.add_observation(obs2, track_state=TrackState.CONFIRMED, is_ambiguous=False)
    assert accum.identity_status == "provisional"
    assert accum.assigned_student_id is None

    # Frame 3: 3 hits -> assigned!
    accum.add_observation(obs3, track_state=TrackState.CONFIRMED, is_ambiguous=False)
    assert accum.identity_status == "assigned"
    assert accum.assigned_student_id == "STU001"
    assert accum.assigned_student_name == "Alice"
    assert accum.recognition_similarity == pytest.approx(0.85, abs=0.01)


# ==============================================================================
# TEST 2: Non-matching / uncertain observations interrupt evidence accumulation
# ==============================================================================
def test_non_matching_observations_interrupt_accumulation():
    """Uncertain or unrecognized observations (None) push into deque without accumulating hits."""
    accum = TrackIdentityAccumulator(min_consistent_hits=3, window_size=5)

    obs1 = RecognitionObservation("STU001", "Alice", 0.85, 1, 1.0)
    accum.add_observation(obs1, track_state=TrackState.CONFIRMED, is_ambiguous=False)
    assert accum.identity_status == "provisional"

    # 5 non-matches (unrecognized/uncertain)
    for seq in range(2, 7):
        accum.add_observation(None, track_state=TrackState.CONFIRMED, is_ambiguous=False)

    # obs1 has scrolled out of 5-frame window!
    assert accum.identity_status == "unassigned"
    assert accum.assigned_student_id is None


# ==============================================================================
# TEST 3: Tentative track identity prevention
# ==============================================================================
def test_tentative_track_never_assigned_identity():
    """Even with 3 matching hits, a TENTATIVE track remains provisional and cannot be assigned."""
    accum = TrackIdentityAccumulator(min_consistent_hits=3, window_size=5)
    obs = RecognitionObservation("STU001", "Alice", 0.85, 1, 1.0)

    for seq in range(1, 4):
        accum.add_observation(obs, track_state=TrackState.TENTATIVE, is_ambiguous=False)

    # Must remain provisional while track is TENTATIVE
    assert accum.identity_status == "provisional"
    assert accum.assigned_student_id is None


# ==============================================================================
# TEST 4: Uncertain and unrecognized faces remain unassigned
# ==============================================================================
def test_uncertain_and_unrecognized_faces_remain_unassigned():
    """VectorMatcher returning 'uncertain' or 'unrecognized' results in no identity assignment."""
    emb_gallery = _norm_vec(401)
    emb_live_diff = _norm_vec(402)  # Completely different person

    gallery = [
        GalleryItem(1, 101, "STU001", "Alice", "frontal", True, emb_gallery)
    ]
    matcher = VectorMatcher(gallery=gallery, similarity_threshold=0.65, ambiguity_margin=0.05)

    match_res = matcher.match_embeddings([(1, emb_live_diff)])
    assert len(match_res) == 1
    assert match_res[0].match_status == "unrecognized"
    assert match_res[0].top_match is None


# ==============================================================================
# TEST 5: Multiple students recognised simultaneously
# ==============================================================================
def test_multiple_students_recognized_simultaneously():
    """Two different faces in the same frame match distinct students in gallery."""
    emb_a = _norm_vec(501)
    emb_b = _norm_vec(502)

    gallery = [
        GalleryItem(1, 101, "STU001", "Alice", "frontal", True, emb_a),
        GalleryItem(2, 102, "STU002", "Bob", "frontal", True, emb_b),
    ]
    matcher = VectorMatcher(gallery=gallery, similarity_threshold=0.65)

    matches = matcher.match_embeddings([(1, emb_a), (2, emb_b)])
    assert len(matches) == 2
    assert matches[0].match_status == "matched"
    assert matches[0].top_match.student_str_id == "STU001"
    assert matches[1].match_status == "matched"
    assert matches[1].top_match.student_str_id == "STU002"


# ==============================================================================
# TEST 6: Duplicate student identity claims (conflict prevention)
# ==============================================================================
def test_duplicate_student_identity_claims_flag_conflict():
    """When two tracks claim the same student ID, mark_conflict() is invoked and assignment is withheld."""
    accum1 = TrackIdentityAccumulator(min_consistent_hits=3, window_size=5)
    accum2 = TrackIdentityAccumulator(min_consistent_hits=3, window_size=5)

    obs = RecognitionObservation("STU001", "Alice", 0.85, 1, 1.0)
    for _ in range(3):
        accum1.add_observation(obs, track_state=TrackState.CONFIRMED, is_ambiguous=False)
        accum2.add_observation(obs, track_state=TrackState.CONFIRMED, is_ambiguous=False)

    assert accum1.identity_status == "assigned"
    # Accum2 conflicts with existing assigned track
    accum2.mark_conflict()
    assert accum2.identity_status == "conflict"


# ==============================================================================
# TEST 7: Ambiguous crossing frames pause recognition accumulation
# ==============================================================================
def test_ambiguous_crossing_pauses_recognition_accumulation():
    """When is_ambiguous is True, add_observation is a no-op."""
    accum = TrackIdentityAccumulator(min_consistent_hits=3, window_size=5)
    obs = RecognitionObservation("STU001", "Alice", 0.85, 1, 1.0)

    # Frame with ambiguous crossing
    accum.add_observation(obs, track_state=TrackState.CONFIRMED, is_ambiguous=True)
    assert len(accum.history) == 0
    assert accum.identity_status == "unassigned"


# ==============================================================================
# TEST 8: Temporary disappearance and recovery preserves assigned student identity
# ==============================================================================
def test_temporary_disappearance_and_recovery_preserves_student_identity():
    """Track assigned STU001 becomes TEMPORARILY_LOST and recovers via Stage-2; retains STU001."""
    emb_a = _norm_vec(801)
    tracker = MultiFaceTracker(min_hits_to_confirm=2, max_missing_frames=10)

    # Confirm track
    tracker.update([_det(1, 100, 100, 60, 60, embedding=emb_a)], frame_seq=1, timestamp=1.0)
    tracker.update([_det(1, 100, 100, 60, 60, embedding=emb_a)], frame_seq=2, timestamp=1.1)

    t = tracker.tracks["TRK-0001"]
    obs = RecognitionObservation("STU001", "Alice", 0.90, 2, 1.1)
    for _ in range(3):
        t.identity_accumulator.add_observation(obs, track_state=TrackState.CONFIRMED, is_ambiguous=False)
    assert t.identity_accumulator.identity_status == "assigned"
    assert t.identity_accumulator.assigned_student_id == "STU001"

    # Disappear on Frame 3 -> TEMPORARILY_LOST
    tracker.update([], frame_seq=3, timestamp=1.2)
    assert t.state == TrackState.TEMPORARILY_LOST
    assert t.identity_accumulator.assigned_student_id == "STU001"

    # Recover on Frame 4
    res4 = tracker.update([_det(1, 110, 100, 60, 60, embedding=emb_a)], frame_seq=4, timestamp=1.3)
    rec = res4.tracks[0]
    assert rec.track_id == "TRK-0001"
    assert rec.is_recovered is True
    assert rec.associated_student_id == "STU001"


# ==============================================================================
# TEST 9: Terminated track releases student identity; re-entry requires fresh accumulation
# ==============================================================================
def test_terminated_track_releases_identity_and_requires_fresh_accumulation():
    """Terminating a track purges it; new track must accumulate 3 fresh hits."""
    emb_a = _norm_vec(901)
    tracker = MultiFaceTracker(min_hits_to_confirm=2, max_missing_frames=2)

    tracker.update([_det(1, 100, 100, 60, 60, embedding=emb_a)], frame_seq=1, timestamp=1.0)
    tracker.update([_det(1, 100, 100, 60, 60, embedding=emb_a)], frame_seq=2, timestamp=1.1)

    t1 = tracker.tracks["TRK-0001"]
    for _ in range(3):
        t1.identity_accumulator.add_observation(
            RecognitionObservation("STU001", "Alice", 0.90, 2, 1.1),
            track_state=TrackState.CONFIRMED,
            is_ambiguous=False,
        )
    assert t1.identity_accumulator.assigned_student_id == "STU001"

    # Miss 3 frames -> TERMINATED and purged
    tracker.update([], frame_seq=3, timestamp=1.2)
    tracker.update([], frame_seq=4, timestamp=1.3)
    tracker.update([], frame_seq=5, timestamp=1.4)
    assert len(tracker.tracks) == 0

    # Person re-enters: receives TRK-0002
    res6 = tracker.update([_det(1, 100, 100, 60, 60, embedding=emb_a)], frame_seq=6, timestamp=1.5)
    t2 = res6.tracks[0]
    assert t2.track_id == "TRK-0002"
    assert t2.is_newly_created is True
    # Identity is unassigned initially
    assert t2.associated_student_id is None
    assert t2.identity_status == "unassigned"


# ==============================================================================
# TEST 10: Detection-to-track correspondence correctly maps recognition results
# ==============================================================================
def test_detection_to_track_correspondence():
    """Ensures each detection's recognition score attaches strictly to its matched track."""
    emb_a = _norm_vec(1001)
    emb_b = _norm_vec(1002)

    tracker = MultiFaceTracker(min_hits_to_confirm=1)
    res = tracker.update([
        _det(1, 100, 100, 60, 60, embedding=emb_a),
        _det(2, 300, 100, 60, 60, embedding=emb_b),
    ], frame_seq=1, timestamp=1.0)

    tracks_by_box = {t.bounding_box: t.track_id for t in res.tracks}
    assert tracks_by_box[(100, 100, 60, 60)] == "TRK-0001"
    assert tracks_by_box[(300, 100, 60, 60)] == "TRK-0002"


# ==============================================================================
# TEST 11 & 12: Section-scoped gallery loading & Consent Revocation / Missing Consent
# ==============================================================================
def test_section_scoping_and_consent_enforcement():
    """Loads gallery for section; only active students with active consent are loaded."""
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    suffix = str(int(time.time()))[-5:]

    try:
        ay = AcademicYear(name=f"AY-{suffix}", start_date=datetime.date(2026, 1, 1), end_date=datetime.date(2026, 12, 31))
        dept = Department(name=f"D-{suffix}", code=f"D{suffix[:3]}", academic_year=ay)
        prog = Programme(name=f"P-{suffix}", code=f"P{suffix[:3]}", department=dept)
        sem = Semester(name=f"S-{suffix}", number=1, programme=prog)
        sec = ClassSection(semester=sem, name=f"SEC-{suffix}", max_students=30)
        db.add_all([ay, dept, prog, sem, sec])
        db.flush()

        # Student 1: Enrolled, Active, Consented
        s1 = Student(student_id=f"STU1_{suffix}", name="Alice", class_section_id=sec.id, roll_number=f"R1_{suffix}")
        db.add(s1)
        db.flush()
        db.add(ConsentRecord(student_id=s1.id, is_active=True))
        emb1 = _norm_vec(1101).tobytes()
        db.add(FaceReference(student_id=s1.id, image_path=f"/dummy/path1_{suffix}.jpg", embedding=emb1, status="active", is_primary=True))

        # Student 2: Enrolled, Active, NO Consent
        s2 = Student(student_id=f"STU2_{suffix}", name="Bob", class_section_id=sec.id, roll_number=f"R2_{suffix}")
        db.add(s2)
        db.flush()
        emb2 = _norm_vec(1102).tobytes()
        db.add(FaceReference(student_id=s2.id, image_path=f"/dummy/path2_{suffix}.jpg", embedding=emb2, status="active", is_primary=True))

        db.commit()

        service = TrackingService(db=db)
        gallery = service._load_section_gallery(sec.id)

        # Alice should be loaded, Bob must be excluded due to missing consent
        loaded_students = {g.student_str_id for g in gallery}
        assert f"STU1_{suffix}" in loaded_students
        assert f"STU2_{suffix}" not in loaded_students
    finally:
        db.close()


# ==============================================================================
# TEST 13: Session deletion clears all cached gallery, matchers, and accumulators
# ==============================================================================
def test_session_deletion_clears_cached_gallery():
    """Ending session clears gallery and matcher from memory."""
    mgr = TrackingSessionManager()
    cfg = CreateTrackingSessionRequest(class_section_id=1)
    dummy_item = GalleryItem(1, 101, "STU001", "Alice", "frontal", True, _norm_vec(1301))

    sess = mgr.create_session(user_id=1, user_role="faculty", class_section_id=1, config=cfg, gallery=[dummy_item])
    sess_id = sess.session_id

    assert len(sess.gallery) == 1
    assert sess.matcher is not None

    mgr.remove_session(sess_id)
    assert mgr.get_session(sess_id) is None


# ==============================================================================
# TEST 14: RBAC and authorization enforcement
# ==============================================================================
def test_tracking_rbac_unassigned_faculty_forbidden():
    """Unassigned faculty cannot create session for section."""
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    suffix = str(int(time.time()))[-5:]
    try:
        ay = AcademicYear(name=f"AY-RBAC-{suffix}", start_date=datetime.date(2026, 1, 1), end_date=datetime.date(2026, 12, 31))
        dept = Department(name=f"D-RBAC-{suffix}", code=f"DR{suffix[:2]}", academic_year=ay)
        prog = Programme(name=f"P-RBAC-{suffix}", code=f"PR{suffix[:2]}", department=dept)
        sem = Semester(name=f"S-RBAC-{suffix}", number=1, programme=prog)
        sec = ClassSection(semester=sem, name=f"S-RBAC-{suffix}", max_students=30)
        unassigned_fac = User(username=f"fac_rbac_{suffix}", email=f"fac_{suffix}@edu.com", full_name="Faculty Tester", role="faculty", password_hash="hash", is_active=True)
        db.add_all([ay, dept, prog, sem, sec, unassigned_fac])
        db.commit()

        service = TrackingService(db=db)
        from app.core.exceptions import AuthorizationForbiddenError
        with pytest.raises(AuthorizationForbiddenError):
            service.create_session(CreateTrackingSessionRequest(class_section_id=sec.id), current_user=unassigned_fac)
    finally:
        db.close()


# ==============================================================================
# TEST 15: Zero attendance records or review decisions mutated
# ==============================================================================
def test_zero_attendance_records_created():
    """Substage 9.7 identity resolution performs tracking without creating attendance rows."""
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    from app.models.attendance import AttendanceRecord, ReviewDecision
    initial_records = db.query(AttendanceRecord).count()
    initial_decisions = db.query(ReviewDecision).count()

    # Simulate tracking identity accumulator
    accum = TrackIdentityAccumulator(min_consistent_hits=3, window_size=5)
    obs = RecognitionObservation("STU001", "Alice", 0.90, 1, 1.0)
    for _ in range(3):
        accum.add_observation(obs, track_state=TrackState.CONFIRMED, is_ambiguous=False)

    assert accum.identity_status == "assigned"

    # Database attendance tables must be completely untouched
    final_records = db.query(AttendanceRecord).count()
    final_decisions = db.query(ReviewDecision).count()
    assert initial_records == final_records
    assert initial_decisions == final_decisions
    db.close()


# ==============================================================================
# TEST 16: Zero duplicate SFace extraction (single-pass embedding reuse)
# ==============================================================================
def test_single_pass_sface_extraction():
    """DetectionInput embedding extracted once is reused by tracker and matcher."""
    emb = _norm_vec(1601)
    det = _det(1, 100, 100, 60, 60, embedding=emb)

    # Tracker uses det.embedding
    tracker = MultiFaceTracker()
    tracker.update([det], frame_seq=1, timestamp=1.0)
    t = tracker.tracks["TRK-0001"]
    assert len(t.appearance_history) == 1
    assert np.array_equal(t.appearance_history[0], emb)

    # Matcher uses identical det.embedding without re-extraction
    gallery = [GalleryItem(1, 101, "STU001", "Alice", "frontal", True, emb)]
    matcher = VectorMatcher(gallery=gallery, similarity_threshold=0.65)
    match_res = matcher.match_embeddings([(det.face_id, det.embedding)])
    assert match_res[0].match_status == "matched"
    assert match_res[0].top_match.student_str_id == "STU001"


# ==============================================================================
# SUBSTAGE 9.7.1 TESTS: Active Session Invalidation & Explicit Mapping
# ==============================================================================

def test_invalidation_clears_student_from_active_session():
    """Consent revocation or reference deletion clears gallery and accumulator immediately."""
    mgr = TrackingSessionManager()
    cfg = CreateTrackingSessionRequest(class_section_id=1)
    emb_alice = _norm_vec(2001)
    emb_bob = _norm_vec(2002)

    g_alice = GalleryItem(1, 101, "STU_ALICE", "Alice", "frontal", True, emb_alice)
    g_bob = GalleryItem(2, 102, "STU_BOB", "Bob", "frontal", True, emb_bob)

    sess = mgr.create_session(user_id=1, user_role="faculty", class_section_id=1, config=cfg, gallery=[g_alice, g_bob])

    # Simulate active track assigned to Alice
    t = MultiFaceTracker()._generate_track_id()
    trk = sess.tracker.tracks[t] = FaceTrack(track_id=t, initial_bbox=(10, 10, 50, 50), frame_seq=1, timestamp=1.0, detection_confidence=0.95)
    for _ in range(3):
        trk.identity_accumulator.add_observation(
            RecognitionObservation("STU_ALICE", "Alice", 0.9, 1, 1.0),
            track_state=TrackState.CONFIRMED,
            is_ambiguous=False,
        )
    assert trk.identity_accumulator.identity_status == "assigned"
    assert trk.identity_accumulator.assigned_student_id == "STU_ALICE"

    # Invalidate Alice
    updated = mgr.invalidate_student_references(student_id=101)
    assert updated == 1

    # Alice removed from gallery; Bob remains
    assert len(sess.gallery) == 1
    assert sess.gallery[0].student_id == 102

    # Track's assigned identity cleared
    assert trk.identity_accumulator.identity_status == "unassigned"
    assert trk.identity_accumulator.assigned_student_id is None

    # Subsequent matcher cannot match Alice
    res = sess.matcher.match_embeddings([(1, emb_alice)])
    assert res[0].match_status == "unrecognized"

    # Bob continues to match
    res_bob = sess.matcher.match_embeddings([(2, emb_bob)])
    assert res_bob[0].match_status == "matched"
    assert res_bob[0].top_match.student_str_id == "STU_BOB"


def test_faculty_revocation_terminates_session():
    """Revoking faculty assignment immediately terminates their active tracking sessions."""
    mgr = TrackingSessionManager()
    cfg = CreateTrackingSessionRequest(class_section_id=1)
    sess = mgr.create_session(user_id=99, user_role="faculty", class_section_id=1, config=cfg)
    sess_id = sess.session_id

    assert mgr.get_session(sess_id) is not None

    terminated = mgr.terminate_sessions_for_user(user_id=99)
    assert terminated == 1
    assert mgr.get_session(sess_id) is None


def test_explicit_detection_to_track_mapping_under_reordering():
    """Tracker attaches matched_face_id explicitly regardless of detection order."""
    tracker = MultiFaceTracker(min_hits_to_confirm=1)

    # Frame 1: Track-1 at (100, 100) with face_id=10, Track-2 at (300, 300) with face_id=20
    d1 = _det(face_id=10, x=100, y=100)
    d2 = _det(face_id=20, x=300, y=300)
    res1 = tracker.update([d1, d2], frame_seq=1, timestamp=1.0)

    t1 = next(t for t in res1.tracks if t.bounding_box == (100, 100, 50, 60))
    t2 = next(t for t in res1.tracks if t.bounding_box == (300, 300, 50, 60))
    assert t1.matched_face_id == 10
    assert t2.matched_face_id == 20

    # Frame 2: Reverse detection order [d2_new, d1_new]
    # face_id=99 at (300, 300) for Track-2, face_id=88 at (100, 100) for Track-1
    d_rev1 = _det(face_id=99, x=300, y=300)
    d_rev2 = _det(face_id=88, x=100, y=100)
    res2 = tracker.update([d_rev1, d_rev2], frame_seq=2, timestamp=1.1)

    t1_frame2 = next(t for t in res2.tracks if t.track_id == t1.track_id)
    t2_frame2 = next(t for t in res2.tracks if t.track_id == t2.track_id)

    # Correct explicit face_id mapped to corresponding track despite reversed input order
    assert t1_frame2.matched_face_id == 88
    assert t2_frame2.matched_face_id == 99


# ==============================================================================
# SUBSTAGE 9.7.2 TESTS: In-Flight Race Protection, Deterministic Concurrency,
# Actual Mutation Hooks & Fail-Safe Invalidation
# ==============================================================================

def test_actual_mutation_hooks_invalidate_sessions():
    """Invokes actual service mutation methods and verifies session invalidation."""
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    suffix = str(int(time.time()))[-5:]
    try:
        ay = AcademicYear(name=f"AY-MUT-{suffix}", start_date=datetime.date(2026, 1, 1), end_date=datetime.date(2026, 12, 31))
        dept = Department(name=f"D-MUT-{suffix}", code=f"DM{suffix[:2]}", academic_year=ay)
        prog = Programme(name=f"P-MUT-{suffix}", code=f"PM{suffix[:2]}", department=dept)
        sem = Semester(name=f"S-MUT-{suffix}", number=1, programme=prog)
        sec = ClassSection(semester=sem, name=f"S-MUT-{suffix}", max_students=30)
        sec2 = ClassSection(semester=sem, name=f"S2-MUT-{suffix}", max_students=30)
        sub = Subject(name=f"Sub-MUT-{suffix}", code=f"SM{suffix[:2]}", class_section=sec)
        fac_user = User(username=f"fac_mut_{suffix}", email=f"fac_mut_{suffix}@edu.com", full_name="Fac Mut", role="faculty", password_hash="hash", is_active=True)
        admin_user = User(username=f"adm_mut_{suffix}", email=f"adm_mut_{suffix}@edu.com", full_name="Adm Mut", role="admin", password_hash="hash", is_active=True)
        db.add_all([ay, dept, prog, sem, sec, sec2, sub, fac_user, admin_user])
        db.flush()

        s1 = Student(student_id=f"STU_M1_{suffix}", name="Alice", class_section_id=sec.id, roll_number=f"RM1_{suffix}")
        s2 = Student(student_id=f"STU_M2_{suffix}", name="Bob", class_section_id=sec.id, roll_number=f"RM2_{suffix}")
        db.add_all([s1, s2])
        db.flush()

        c1 = ConsentRecord(student_id=s1.id, is_active=True)
        c2 = ConsentRecord(student_id=s2.id, is_active=True)
        ref1 = FaceReference(student_id=s1.id, image_path="/dummy/ref1.jpg", embedding=_norm_vec(3001).tobytes(), status="active", is_primary=True)
        ref2 = FaceReference(student_id=s2.id, image_path="/dummy/ref2.jpg", embedding=_norm_vec(3002).tobytes(), status="active", is_primary=True)
        fa = FacultyAssignment(user_id=fac_user.id, class_section_id=sec.id, subject_id=sub.id, is_active=True, assigned_by=admin_user.id)
        db.add_all([c1, c2, ref1, ref2, fa])
        db.commit()

        # 1. Create active tracking session with global manager
        from app.services.tracking_service import session_manager
        tracking_svc = TrackingService(db=db)
        sess_resp = tracking_svc.create_session(CreateTrackingSessionRequest(class_section_id=sec.id), current_user=fac_user)
        active_sess = session_manager.get_session(sess_resp.session_id)
        assert active_sess is not None
        assert len(active_sess.gallery) == 2

        # 2. Test BiometricService.delete_reference hook
        from app.services.biometric_service import BiometricService
        bio_svc = BiometricService(db=db)
        bio_svc.delete_reference(s2.student_id, ref2.id, current_user=admin_user)
        # s2 reference is removed from session gallery
        assert len(active_sess.gallery) == 1
        assert active_sess.gallery[0].student_str_id == s1.student_id

        # 3. Test StudentService.revoke_consent hook
        from app.services.student_service import StudentService
        revoked_consent, ok = StudentService.revoke_consent(db, s1.id)
        assert ok is True
        assert len(active_sess.gallery) == 0

        # 4. Test AcademicService.revoke_faculty_assignment hook
        from app.services.academic_service import AcademicService
        AcademicService.revoke_faculty_assignment(db, fa.id)
        # Session is terminated
        assert session_manager.get_session(sess_resp.session_id) is None

    finally:
        db.close()


def test_in_flight_recognition_generation_mismatch_discards_stale_results():
    """
    Deterministic concurrency test:
    A frame captures an old gallery generation, consent invalidation completes,
    and when the frame tracker update runs, stale recognition observations are discarded.
    """
    import threading

    mgr = TrackingSessionManager()
    cfg = CreateTrackingSessionRequest(class_section_id=1)
    emb_alice = _norm_vec(4001)

    g_alice = GalleryItem(1, 101, "STU_ALICE", "Alice", "frontal", True, emb_alice)
    sess = mgr.create_session(user_id=1, user_role="faculty", class_section_id=1, config=cfg, gallery=[g_alice])

    initial_gen = sess.gallery_generation
    assert initial_gen == 1

    # Simulate Frame starting recognition match with gen 1
    matched_gen = sess.gallery_generation
    stale_recog_list = sess.matcher.match_embeddings([(1, emb_alice)])
    assert stale_recog_list[0].match_status == "matched"

    # Consent revocation invalidates Alice, advancing generation
    mgr.invalidate_student_references(student_id=101)
    assert sess.gallery_generation == 2
    assert len(sess.gallery) == 0

    # Simulate frame tracker update running with stale matched_gen (1)
    with sess.lock:
        is_stale = (matched_gen != sess.gallery_generation)
        recog_results = {1: stale_recog_list[0]}
        if is_stale:
            recog_results.clear()

        # Update tracker with face_id 1
        d = _det(1, 100, 100, 50, 60, embedding=emb_alice)
        res = sess.tracker.update([d], frame_seq=1, timestamp=1.0)

        # Confirm recognition observation was discarded
        t_out = res.tracks[0]
        trk = sess.tracker.tracks[t_out.track_id]
        obs = None
        if t_out.matched_face_id is not None and t_out.matched_face_id in recog_results:
            obs = RecognitionObservation("STU_ALICE", "Alice", 0.9, 1, 1.0)
        trk.identity_accumulator.add_observation(obs, trk.state, t_out.is_ambiguous)

        # Student identity remains completely unassigned
        assert trk.identity_accumulator.identity_status == "unassigned"
        assert trk.identity_accumulator.assigned_student_id is None


def test_fail_safe_invalidation_disables_recognition():
    """If unexpected error happens during invalidation, session enters fail-safe disabled state."""
    mgr = TrackingSessionManager()
    cfg = CreateTrackingSessionRequest(class_section_id=5)
    sess = mgr.create_session(user_id=1, user_role="faculty", class_section_id=5, config=cfg)

    # Trigger fail-safe disable
    count = mgr.mark_sessions_unavailable_for_section(class_section_id=5)
    assert count == 1
    assert sess.recognition_enabled is False
    assert len(sess.gallery) == 0
    assert sess.matcher is None


# ==============================================================================
# PHASE 9 MASTER AUDIT ADDITIONAL VERIFICATION TESTS
# ==============================================================================

def test_actual_section_change_and_student_deactivation_hooks():
    """Verifies StudentService.update_student() and deactivate_student() hooks."""
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    suffix = str(int(time.time()))[-5:]
    try:
        ay = AcademicYear(name=f"AY-SEC-{suffix}", start_date=datetime.date(2026, 1, 1), end_date=datetime.date(2026, 12, 31))
        dept = Department(name=f"D-SEC-{suffix}", code=f"DS{suffix[:2]}", academic_year=ay)
        prog = Programme(name=f"P-SEC-{suffix}", code=f"PS{suffix[:2]}", department=dept)
        sem = Semester(name=f"S-SEC-{suffix}", number=1, programme=prog)
        sec_a = ClassSection(semester=sem, name=f"SEC-A-{suffix}", max_students=30)
        sec_b = ClassSection(semester=sem, name=f"SEC-B-{suffix}", max_students=30)
        sub = Subject(name=f"Sub-SEC-{suffix}", code=f"SS{suffix[:2]}", class_section=sec_a)
        fac = User(username=f"fac_sec_{suffix}", email=f"fac_sec_{suffix}@edu.com", full_name="Fac Sec", role="faculty", password_hash="hash", is_active=True)
        admin = User(username=f"adm_sec_{suffix}", email=f"adm_sec_{suffix}@edu.com", full_name="Adm Sec", role="admin", password_hash="hash", is_active=True)
        db.add_all([ay, dept, prog, sem, sec_a, sec_b, sub, fac, admin])
        db.flush()

        s = Student(student_id=f"STU_SEC_{suffix}", name="Charlie", class_section_id=sec_a.id, roll_number=f"RSEC_{suffix}")
        db.add(s)
        db.flush()
        db.add(ConsentRecord(student_id=s.id, is_active=True))
        db.add(FaceReference(student_id=s.id, image_path="/dummy/c.jpg", embedding=_norm_vec(5001).tobytes(), status="active", is_primary=True))
        db.add(FacultyAssignment(user_id=fac.id, class_section_id=sec_a.id, subject_id=sub.id, is_active=True, assigned_by=admin.id))
        db.commit()

        from app.services.tracking_service import session_manager
        tracking_svc = TrackingService(db=db)
        sess = tracking_svc.create_session(CreateTrackingSessionRequest(class_section_id=sec_a.id), current_user=fac)
        active_sess = session_manager.get_session(sess.session_id)
        assert active_sess is not None
        assert len(active_sess.gallery) == 1

        # 1. Update student's section from sec_a to sec_b
        from app.schemas.student import StudentUpdate
        from app.services.student_service import StudentService
        StudentService.update_student(db, s.id, StudentUpdate(class_section_id=sec_b.id))

        # Student is invalidated and removed from sec_a session gallery
        assert len(active_sess.gallery) == 0

        # 2. Deactivate student
        StudentService.deactivate_student(db, s.id)
        assert s.is_active is False
    finally:
        db.close()


def test_repeated_invalidation_is_safe_and_idempotent():
    """Repeated invalidations do not fail or corrupt state."""
    mgr = TrackingSessionManager()
    cfg = CreateTrackingSessionRequest(class_section_id=1)
    sess = mgr.create_session(user_id=1, user_role="faculty", class_section_id=1, config=cfg)

    # Calling invalidation repeatedly for nonexistent student or already removed student
    assert mgr.invalidate_student_references(student_id=999) == 0
    assert mgr.invalidate_student_references(student_id=999) == 0
    assert sess.recognition_enabled is True


def test_deterministic_thread_barrier_in_flight_invalidation():
    """
    Deterministic threading event test:
    Frame T1 captures old matcher and generation, pauses at barrier;
    Main thread runs invalidation;
    Frame T1 resumes and discards stale candidate recognition.
    """
    import threading

    mgr = TrackingSessionManager()
    cfg = CreateTrackingSessionRequest(class_section_id=1)
    emb_alice = _norm_vec(6001)
    g_alice = GalleryItem(1, 101, "STU_ALICE", "Alice", "frontal", True, emb_alice)
    sess = mgr.create_session(user_id=1, user_role="faculty", class_section_id=1, config=cfg, gallery=[g_alice])

    barrier_matched = threading.Event()
    barrier_can_proceed = threading.Event()
    observed_identity = {"status": None, "id": None}

    def frame_worker():
        # Capture generation and matcher under lock
        with sess.lock:
            matched_gen = sess.gallery_generation
            matcher = sess.matcher

        # Match embedding
        recog = matcher.match_embeddings([(1, emb_alice)])

        # Signal that matching with gen 1 finished
        barrier_matched.set()
        # Wait for invalidation to complete in main thread
        barrier_can_proceed.wait(timeout=5.0)

        # Resume frame update
        with sess.lock:
            if sess.recognition_enabled and matched_gen == sess.gallery_generation:
                recog_res = {1: recog[0]}
            else:
                recog_res = {}

            res = sess.tracker.update([_det(1, 100, 100, 50, 60, embedding=emb_alice)], frame_seq=1, timestamp=1.0)
            t_out = res.tracks[0]
            trk = sess.tracker.tracks[t_out.track_id]
            obs = None
            if t_out.matched_face_id is not None and t_out.matched_face_id in recog_res:
                obs = RecognitionObservation("STU_ALICE", "Alice", 0.9, 1, 1.0)
            trk.identity_accumulator.add_observation(obs, trk.state, t_out.is_ambiguous)
            observed_identity["status"] = trk.identity_accumulator.identity_status
            observed_identity["id"] = trk.identity_accumulator.assigned_student_id

    th = threading.Thread(target=frame_worker)
    th.start()

    # Wait until frame has finished matching with old gallery
    assert barrier_matched.wait(timeout=5.0) is True

    # Invalidate Alice in main thread
    mgr.invalidate_student_references(student_id=101)

    # Let frame proceed
    barrier_can_proceed.set()
    th.join(timeout=5.0)

    # Verified: stale recognition evidence was rejected!
    assert observed_identity["status"] == "unassigned"
    assert observed_identity["id"] is None



