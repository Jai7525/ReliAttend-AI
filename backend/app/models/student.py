"""
ReliAttend AI — Student, Consent, and Biometric Face Reference Models
Handles student profiles, biometric consent metadata, and secure face embeddings.
"""

from datetime import datetime, timezone
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship
from app.database import Base


class Student(Base):
    __tablename__ = "students"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    student_id = Column(String(50), unique=True, index=True, nullable=False)  # Institutional ID (e.g. "STU2026001")
    name = Column(String(200), nullable=False)
    roll_number = Column(String(50), nullable=False)
    class_section_id = Column(Integer, ForeignKey("class_sections.id", ondelete="RESTRICT"), nullable=False)
    registration_status = Column(String(20), default="pending", nullable=False)  # pending, enrolled, deactivated
    is_active = Column(Boolean, default=True, nullable=False)
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

    __table_args__ = (
        UniqueConstraint("roll_number", "class_section_id", name="uq_roll_number_section"),
    )

    # Relationships
    class_section = relationship("ClassSection", back_populates="students")
    consent_record = relationship("ConsentRecord", back_populates="student", uselist=False, cascade="all, delete-orphan")
    face_reference = relationship("FaceReference", back_populates="student", uselist=False, cascade="all, delete-orphan")
    attendance_records = relationship("AttendanceRecord", back_populates="student")
    detection_evidences = relationship("DetectionEvidence", back_populates="candidate_student")


class ConsentRecord(Base):
    """
    Biometric Consent & Authorization (BR-018, FR-ADMIN-004):
    Guarantees student biometric enrolment is blocked unless valid consent metadata is stored.
    """
    __tablename__ = "consent_records"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    student_id = Column(Integer, ForeignKey("students.id", ondelete="CASCADE"), unique=True, nullable=False)
    consent_type = Column(String(50), default="institutional_enrollment", nullable=False)
    consent_reference = Column(String(200), nullable=True)  # Signed form ID or approval reference
    granted_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    granted_by = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)

    # Relationships
    student = relationship("Student", back_populates="consent_record")
    recorded_by = relationship("User")


class FaceReference(Base):
    """
    Sensitive Biometric Face Reference (BR-017, BR-019, FR-STU-005):
    Stores face embeddings as binary BLOB (LargeBinary) and points to protected reference images.
    Always maintains model name, model version, and quality score.
    """
    __tablename__ = "face_references"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    student_id = Column(Integer, ForeignKey("students.id", ondelete="CASCADE"), unique=True, nullable=False)
    image_path = Column(String(500), nullable=False)  # Secure storage path
    embedding = Column(LargeBinary, nullable=False)  # Serialized numpy float array
    embedding_size = Column(Integer, default=512, nullable=False)
    model_name = Column(String(100), default="insightface_buffalo_l", nullable=False)
    model_version = Column(String(50), default="1.0", nullable=False)
    quality_score = Column(Float, nullable=True)  # Quality score at time of enrollment
    status = Column(String(20), default="active", nullable=False)  # active, deactivated, deleted
    enrolled_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    enrolled_by = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    deactivated_at = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    student = relationship("Student", back_populates="face_reference")
    enroller = relationship("User")

