"""
ReliAttend AI — Attendance Session, Evidence, Records, Review, and Audit Models
Implements duplicate prevention (BR-012), model version traceability (BR-019),
append-only reviews/corrections (BR-011, BR-015), and audit logging (BR-022).
"""

from datetime import datetime, timezone
from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship
from app.database import Base


class ModelConfig(Base):
    """
    AI Model & Threshold Configuration (BR-019, FR-ADMIN-003):
    Records the specific versions and weights used during an attendance session.
    """
    __tablename__ = "model_configs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    detection_model = Column(String(100), default="scrfd_2.5g", nullable=False)
    detection_version = Column(String(50), default="1.0", nullable=False)
    recognition_model = Column(String(100), default="arcface_r50", nullable=False)
    recognition_version = Column(String(50), default="1.0", nullable=False)
    recognition_threshold = Column(Float, default=0.60, nullable=False)
    pad_model = Column(String(100), default="silent_face_anti_spoofing", nullable=True)
    pad_version = Column(String(50), default="1.0", nullable=True)
    pad_threshold = Column(Float, default=0.70, nullable=True)
    reliability_weights = Column(
        Text,
        default='{"wC": 0.40, "wQ": 0.20, "wT": 0.20, "wL": 0.20}',
        nullable=False,
    )
    reliability_threshold = Column(Float, default=0.75, nullable=False)
    min_frames_required = Column(Integer, default=5, nullable=False)  # Multi-frame evidence (BR-007)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relationships
    sessions = relationship("AttendanceSession", back_populates="model_config")


class AttendanceSession(Base):
    """
    Attendance Session Lifecycle (BR-002, BR-014, FR-SES-004):
    Every session links to one class, subject, period, faculty, and date.
    """
    __tablename__ = "attendance_sessions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    faculty_id = Column(Integer, ForeignKey("users.id", ondelete="RESTRICT"), nullable=False)
    class_section_id = Column(Integer, ForeignKey("class_sections.id", ondelete="RESTRICT"), nullable=False)
    subject_id = Column(Integer, ForeignKey("subjects.id", ondelete="RESTRICT"), nullable=False)
    period_id = Column(Integer, ForeignKey("periods.id", ondelete="RESTRICT"), nullable=False)
    session_date = Column(Date, nullable=False)
    start_time = Column(DateTime(timezone=True), nullable=False)
    end_time = Column(DateTime(timezone=True), nullable=True)
    status = Column(String(20), default="active", nullable=False)  # active, completed, cancelled
    model_config_id = Column(Integer, ForeignKey("model_configs.id", ondelete="SET NULL"), nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relationships
    faculty = relationship("User", back_populates="sessions")
    class_section = relationship("ClassSection", back_populates="attendance_sessions")
    subject = relationship("Subject", back_populates="attendance_sessions")
    period = relationship("Period", back_populates="attendance_sessions")
    model_config = relationship("ModelConfig", back_populates="sessions")
    detection_evidences = relationship("DetectionEvidence", back_populates="session", cascade="all, delete-orphan")
    attendance_records = relationship("AttendanceRecord", back_populates="session", cascade="all, delete-orphan")


class DetectionEvidence(Base):
    """
    Per-track Multi-Frame Evidence (BR-007, FR-TRK-001, FR-REL-001):
    Accumulates observations over time without permanently saving raw video frames (BR-016).
    """
    __tablename__ = "detection_evidence"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("attendance_sessions.id", ondelete="CASCADE"), nullable=False)
    track_id = Column(String(50), nullable=False)  # Temporary tracking identifier
    candidate_student_id = Column(Integer, ForeignKey("students.id", ondelete="SET NULL"), nullable=True)
    best_match_score = Column(Float, nullable=True)
    avg_match_score = Column(Float, nullable=True)
    frames_detected = Column(Integer, default=0, nullable=False)
    frames_recognised = Column(Integer, default=0, nullable=False)
    avg_quality_score = Column(Float, nullable=True)
    temporal_consistency = Column(Float, nullable=True)
    pad_result = Column(String(20), nullable=True)  # genuine, suspicious, uncertain
    pad_score = Column(Float, nullable=True)
    reliability_score = Column(Float, nullable=True)
    decision_state = Column(String(20), default="pending", nullable=False)  # confirmed, uncertain, unknown, suspicious
    reason_codes = Column(Text, nullable=True)  # JSON array string e.g. '["weak_match", "pad_suspicious"]'
    first_seen_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    last_seen_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relationships
    session = relationship("AttendanceSession", back_populates="detection_evidences")
    candidate_student = relationship("Student", back_populates="detection_evidences")
    attendance_record = relationship("AttendanceRecord", back_populates="evidence", uselist=False)


class AttendanceRecord(Base):
    """
    Final Student Attendance Record per Session (BR-012, FR-ATT-001):
    Enforces AT MOST ONE record per student per session via UniqueConstraint.
    """
    __tablename__ = "attendance_records"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("attendance_sessions.id", ondelete="CASCADE"), nullable=False)
    student_id = Column(Integer, ForeignKey("students.id", ondelete="RESTRICT"), nullable=False)
    status = Column(String(20), nullable=False)  # confirmed, pending_review, unknown, absent
    source = Column(String(20), default="auto", nullable=False)  # auto, manual, review
    reliability_score = Column(Float, nullable=True)
    evidence_id = Column(Integer, ForeignKey("detection_evidence.id", ondelete="SET NULL"), nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Strict Database Integrity (BR-012): One record per student per session
    __table_args__ = (
        UniqueConstraint("session_id", "student_id", name="uq_session_student_attendance"),
    )

    # Relationships
    session = relationship("AttendanceSession", back_populates="attendance_records")
    student = relationship("Student", back_populates="attendance_records")
    evidence = relationship("DetectionEvidence", back_populates="attendance_record")
    review_decisions = relationship("ReviewDecision", back_populates="attendance_record", cascade="all, delete-orphan")
    corrections = relationship("AttendanceCorrection", back_populates="attendance_record", cascade="all, delete-orphan")


class ReviewDecision(Base):
    """
    Faculty Review of Uncertain/Suspicious Matches (BR-011, BR-022, FR-ATT-003):
    Append-only log of faculty decisions with timestamps and mandatory reasons.
    """
    __tablename__ = "review_decisions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    attendance_record_id = Column(Integer, ForeignKey("attendance_records.id", ondelete="CASCADE"), nullable=False)
    reviewer_id = Column(Integer, ForeignKey("users.id", ondelete="RESTRICT"), nullable=False)
    decision = Column(String(20), nullable=False)  # confirmed, rejected, pending
    reason = Column(Text, nullable=True)
    decided_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relationships
    attendance_record = relationship("AttendanceRecord", back_populates="review_decisions")
    reviewer = relationship("User", back_populates="review_decisions")


class AttendanceCorrection(Base):
    """
    Manual Attendance Correction Audit Trail (BR-015, BR-022, FR-ATT-004):
    Tracks actor, timestamp, previous status, new status, and reason for any change.
    """
    __tablename__ = "attendance_corrections"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    attendance_record_id = Column(Integer, ForeignKey("attendance_records.id", ondelete="CASCADE"), nullable=False)
    corrected_by = Column(Integer, ForeignKey("users.id", ondelete="RESTRICT"), nullable=False)
    previous_status = Column(String(20), nullable=False)
    new_status = Column(String(20), nullable=False)
    reason = Column(Text, nullable=False)  # Mandatory explanation for audit
    corrected_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relationships
    attendance_record = relationship("AttendanceRecord", back_populates="corrections")
    corrected_by_user = relationship("User", back_populates="corrections")


class AuditLog(Base):
    """
    System-wide Audit Logging (BR-022, FR-AUDIT-001):
    Immutable log of critical actions (enrolment, session start/end, review, correction, deletion).
    """
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    actor_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    action = Column(String(100), nullable=False)  # e.g. "student.enrolled", "session.started"
    target_type = Column(String(50), nullable=True)  # e.g. "Student", "AttendanceSession"
    target_id = Column(Integer, nullable=True)
    details = Column(Text, nullable=True)  # JSON metadata payload
    ip_address = Column(String(45), nullable=True)
    timestamp = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relationships
    actor = relationship("User", back_populates="audit_logs")

