"""
ReliAttend AI — Database Models Aggregator
Exports Base and all SQLAlchemy ORM models.
"""

from app.database import Base

# User Model
from app.models.user import User

# Academic Hierarchy Models
from app.models.academic import (
    AcademicYear,
    Department,
    Programme,
    Semester,
    ClassSection,
    Subject,
    Period,
    FacultyAssignment,
)

# Student & Biometric Models
from app.models.student import (
    Student,
    ConsentRecord,
    FaceReference,
)

# Attendance Session, Evidence, Review & Audit Models
from app.models.attendance import (
    ModelConfig,
    AttendanceSession,
    DetectionEvidence,
    AttendanceRecord,
    ReviewDecision,
    AttendanceCorrection,
    AuditLog,
)

__all__ = [
    "Base",
    "User",
    "AcademicYear",
    "Department",
    "Programme",
    "Semester",
    "ClassSection",
    "Subject",
    "Period",
    "FacultyAssignment",
    "Student",
    "ConsentRecord",
    "FaceReference",
    "ModelConfig",
    "AttendanceSession",
    "DetectionEvidence",
    "AttendanceRecord",
    "ReviewDecision",
    "AttendanceCorrection",
    "AuditLog",
]
