"""
ReliAttend AI — Schemas Package Aggregator
"""

from app.schemas.auth import (
    LoginRequest,
    RefreshTokenRequest,
    TokenResponse,
    UserResponse,
)

from app.schemas.academic import (
    AcademicYearCreate,
    AcademicYearUpdate,
    AcademicYearResponse,
    DepartmentCreate,
    DepartmentUpdate,
    DepartmentResponse,
    ProgrammeCreate,
    ProgrammeResponse,
    SemesterCreate,
    SemesterResponse,
    ClassSectionCreate,
    ClassSectionUpdate,
    ClassSectionResponse,
    SubjectCreate,
    SubjectResponse,
    PeriodCreate,
    PeriodResponse,
    FacultyAssignmentCreate,
    FacultyAssignmentResponse,
    AssignedClassDetail,
)

__all__ = [
    "LoginRequest",
    "RefreshTokenRequest",
    "TokenResponse",
    "UserResponse",
    "AcademicYearCreate",
    "AcademicYearUpdate",
    "AcademicYearResponse",
    "DepartmentCreate",
    "DepartmentUpdate",
    "DepartmentResponse",
    "ProgrammeCreate",
    "ProgrammeResponse",
    "SemesterCreate",
    "SemesterResponse",
    "ClassSectionCreate",
    "ClassSectionUpdate",
    "ClassSectionResponse",
    "SubjectCreate",
    "SubjectResponse",
    "PeriodCreate",
    "PeriodResponse",
    "FacultyAssignmentCreate",
    "FacultyAssignmentResponse",
    "AssignedClassDetail",
]
