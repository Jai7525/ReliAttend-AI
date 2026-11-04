"""
ReliAttend AI — Academic Hierarchy Models
Includes AcademicYear, Department, Programme, Semester, ClassSection,
Subject, Period, and FacultyAssignment.
"""

from datetime import datetime, timezone
from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Time,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship
from app.database import Base


class AcademicYear(Base):
    __tablename__ = "academic_years"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String(50), unique=True, nullable=False)  # e.g. "2026-2027"
    start_date = Column(Date, nullable=False)
    end_date = Column(Date, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relationships
    departments = relationship("Department", back_populates="academic_year", cascade="all, delete-orphan")


class Department(Base):
    __tablename__ = "departments"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String(200), nullable=False)  # e.g. "Computer Science & Engineering"
    code = Column(String(20), nullable=False)  # e.g. "CSE"
    academic_year_id = Column(Integer, ForeignKey("academic_years.id", ondelete="CASCADE"), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint("code", "academic_year_id", name="uq_dept_code_year"),
    )

    # Relationships
    academic_year = relationship("AcademicYear", back_populates="departments")
    programmes = relationship("Programme", back_populates="department", cascade="all, delete-orphan")


class Programme(Base):
    __tablename__ = "programmes"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String(200), nullable=False)  # e.g. "B.Tech in Artificial Intelligence"
    code = Column(String(20), nullable=False)  # e.g. "BTECH-AI"
    department_id = Column(Integer, ForeignKey("departments.id", ondelete="CASCADE"), nullable=False)
    duration_years = Column(Integer, default=4, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint("code", "department_id", name="uq_prog_code_dept"),
    )

    # Relationships
    department = relationship("Department", back_populates="programmes")
    semesters = relationship("Semester", back_populates="programme", cascade="all, delete-orphan")


class Semester(Base):
    __tablename__ = "semesters"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String(50), nullable=False)  # e.g. "Semester 5"
    number = Column(Integer, nullable=False)  # 1 through 8
    programme_id = Column(Integer, ForeignKey("programmes.id", ondelete="CASCADE"), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint("number", "programme_id", name="uq_sem_number_prog"),
    )

    # Relationships
    programme = relationship("Programme", back_populates="semesters")
    class_sections = relationship("ClassSection", back_populates="semester", cascade="all, delete-orphan")


class ClassSection(Base):
    __tablename__ = "class_sections"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String(100), nullable=False)  # e.g. "Section A"
    semester_id = Column(Integer, ForeignKey("semesters.id", ondelete="CASCADE"), nullable=False)
    max_students = Column(Integer, nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint("name", "semester_id", name="uq_section_name_sem"),
    )

    # Relationships
    semester = relationship("Semester", back_populates="class_sections")
    subjects = relationship("Subject", back_populates="class_section", cascade="all, delete-orphan")
    students = relationship("Student", back_populates="class_section")
    periods = relationship("Period", back_populates="class_section")
    faculty_assignments = relationship("FacultyAssignment", back_populates="class_section")
    attendance_sessions = relationship("AttendanceSession", back_populates="class_section")


class Subject(Base):
    __tablename__ = "subjects"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String(200), nullable=False)  # e.g. "Computer Vision and Deep Learning"
    code = Column(String(20), nullable=False)  # e.g. "CS501"
    class_section_id = Column(Integer, ForeignKey("class_sections.id", ondelete="CASCADE"), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint("code", "class_section_id", name="uq_sub_code_section"),
    )

    # Relationships
    class_section = relationship("ClassSection", back_populates="subjects")
    periods = relationship("Period", back_populates="subject")
    faculty_assignments = relationship("FacultyAssignment", back_populates="subject")
    attendance_sessions = relationship("AttendanceSession", back_populates="subject")


class Period(Base):
    __tablename__ = "periods"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String(50), nullable=False)  # e.g. "Period 1"
    day_of_week = Column(String(10), nullable=False)  # Monday, Tuesday, etc.
    start_time = Column(Time, nullable=False)
    end_time = Column(Time, nullable=False)
    subject_id = Column(Integer, ForeignKey("subjects.id", ondelete="CASCADE"), nullable=False)
    class_section_id = Column(Integer, ForeignKey("class_sections.id", ondelete="CASCADE"), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relationships
    subject = relationship("Subject", back_populates="periods")
    class_section = relationship("ClassSection", back_populates="periods")
    attendance_sessions = relationship("AttendanceSession", back_populates="period")


class FacultyAssignment(Base):
    """
    Core Authorization Entity (BR-001):
    Maps a faculty user to a specific class and subject.
    Only assigned faculty can start an attendance session.
    """
    __tablename__ = "faculty_assignments"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    class_section_id = Column(Integer, ForeignKey("class_sections.id", ondelete="CASCADE"), nullable=False)
    subject_id = Column(Integer, ForeignKey("subjects.id", ondelete="CASCADE"), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    assigned_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    assigned_by = Column(Integer, ForeignKey("users.id"), nullable=False)

    __table_args__ = (
        UniqueConstraint("user_id", "class_section_id", "subject_id", name="uq_faculty_class_subject"),
    )

    # Relationships
    faculty = relationship("User", back_populates="faculty_assignments", foreign_keys=[user_id])
    assigned_by_user = relationship("User", back_populates="assigned_by_faculty", foreign_keys=[assigned_by])
    class_section = relationship("ClassSection", back_populates="faculty_assignments")
    subject = relationship("Subject", back_populates="faculty_assignments")

