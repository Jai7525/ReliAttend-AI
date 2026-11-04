"""
ReliAttend AI — Academic Hierarchy Pydantic Schemas
Validates payloads for years, departments, programmes, semesters,
classes, subjects, periods, and faculty assignments.
"""

from datetime import date, datetime, time
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field


# ----------------- Academic Year -----------------
class AcademicYearCreate(BaseModel):
    name: str = Field(..., examples=["2026-2027"])
    start_date: date = Field(..., examples=["2026-08-01"])
    end_date: date = Field(..., examples=["2027-05-31"])
    is_active: bool = True


class AcademicYearUpdate(BaseModel):
    name: Optional[str] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    is_active: Optional[bool] = None


class AcademicYearResponse(BaseModel):
    id: int
    name: str
    start_date: date
    end_date: date
    is_active: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ----------------- Department -----------------
class DepartmentCreate(BaseModel):
    name: str = Field(..., examples=["Computer Science & Engineering"])
    code: str = Field(..., examples=["CSE"])
    academic_year_id: int = Field(...)
    is_active: bool = True


class DepartmentUpdate(BaseModel):
    name: Optional[str] = None
    code: Optional[str] = None
    is_active: Optional[bool] = None


class DepartmentResponse(BaseModel):
    id: int
    name: str
    code: str
    academic_year_id: int
    is_active: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ----------------- Programme -----------------
class ProgrammeCreate(BaseModel):
    name: str = Field(..., examples=["B.Tech in Artificial Intelligence"])
    code: str = Field(..., examples=["BTECH-AI"])
    department_id: int = Field(...)
    duration_years: int = Field(default=4, ge=1, le=6)
    is_active: bool = True


class ProgrammeResponse(BaseModel):
    id: int
    name: str
    code: str
    department_id: int
    duration_years: int
    is_active: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ----------------- Semester -----------------
class SemesterCreate(BaseModel):
    name: str = Field(..., examples=["Semester 5"])
    number: int = Field(..., ge=1, le=12, examples=[5])
    programme_id: int = Field(...)
    is_active: bool = True


class SemesterResponse(BaseModel):
    id: int
    name: str
    number: int
    programme_id: int
    is_active: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ----------------- Class Section -----------------
class ClassSectionCreate(BaseModel):
    name: str = Field(..., examples=["Section A"])
    semester_id: int = Field(...)
    max_students: Optional[int] = Field(default=60, ge=1)
    is_active: bool = True


class ClassSectionUpdate(BaseModel):
    name: Optional[str] = None
    max_students: Optional[int] = None
    is_active: Optional[bool] = None


class ClassSectionResponse(BaseModel):
    id: int
    name: str
    semester_id: int
    max_students: Optional[int]
    is_active: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ----------------- Subject -----------------
class SubjectCreate(BaseModel):
    name: str = Field(..., examples=["Computer Vision & Deep Learning"])
    code: str = Field(..., examples=["CS501"])
    class_section_id: int = Field(...)
    is_active: bool = True


class SubjectResponse(BaseModel):
    id: int
    name: str
    code: str
    class_section_id: int
    is_active: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ----------------- Period -----------------
class PeriodCreate(BaseModel):
    name: str = Field(..., examples=["Period 1"])
    day_of_week: str = Field(..., examples=["Monday"])
    start_time: time = Field(..., examples=["09:00:00"])
    end_time: time = Field(..., examples=["09:50:00"])
    subject_id: int = Field(...)
    class_section_id: int = Field(...)
    is_active: bool = True


class PeriodResponse(BaseModel):
    id: int
    name: str
    day_of_week: str
    start_time: time
    end_time: time
    subject_id: int
    class_section_id: int
    is_active: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ----------------- Faculty Assignment -----------------
class FacultyAssignmentCreate(BaseModel):
    user_id: int = Field(..., description="Faculty user account ID")
    class_section_id: int = Field(...)
    subject_id: int = Field(...)
    is_active: bool = True


class FacultyAssignmentResponse(BaseModel):
    id: int
    user_id: int
    class_section_id: int
    subject_id: int
    is_active: bool
    assigned_at: datetime
    assigned_by: int

    model_config = ConfigDict(from_attributes=True)


class AssignedClassDetail(BaseModel):
    assignment_id: int
    class_section_id: int
    class_section_name: str
    subject_id: int
    subject_name: str
    subject_code: str

