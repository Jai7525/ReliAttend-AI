"""
ReliAttend AI — User Model
Represents administrator and faculty user accounts.
"""

from datetime import datetime, timezone
from sqlalchemy import Boolean, Column, DateTime, Integer, String, CheckConstraint
from sqlalchemy.orm import relationship
from app.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(100), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    full_name = Column(String(200), nullable=False)
    role = Column(
        String(20),
        CheckConstraint("role IN ('admin', 'faculty')", name="valid_user_role"),
        nullable=False,
        default="faculty",
    )
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

    # Relationships
    faculty_assignments = relationship("FacultyAssignment", back_populates="faculty", foreign_keys="FacultyAssignment.user_id")
    assigned_by_faculty = relationship("FacultyAssignment", back_populates="assigned_by_user", foreign_keys="FacultyAssignment.assigned_by")
    sessions = relationship("AttendanceSession", back_populates="faculty")
    review_decisions = relationship("ReviewDecision", back_populates="reviewer")
    corrections = relationship("AttendanceCorrection", back_populates="corrected_by_user")
    audit_logs = relationship("AuditLog", back_populates="actor")

    def __repr__(self) -> str:
        return f"<User(id={self.id}, username='{self.username}', role='{self.role}')>"

