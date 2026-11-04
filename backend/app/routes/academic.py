"""
ReliAttend AI — Academic Management Endpoints Router
Exposes CRUD operations for educational structure and faculty assignments.
"""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.user import User
from app.routes.deps import get_current_user, require_admin, require_faculty
from app.schemas.academic import (
    AcademicYearCreate,
    AcademicYearResponse,
    AcademicYearUpdate,
    AssignedClassDetail,
    ClassSectionCreate,
    ClassSectionResponse,
    ClassSectionUpdate,
    DepartmentCreate,
    DepartmentResponse,
    DepartmentUpdate,
    FacultyAssignmentCreate,
    FacultyAssignmentResponse,
    PeriodCreate,
    PeriodResponse,
    ProgrammeCreate,
    ProgrammeResponse,
    SemesterCreate,
    SemesterResponse,
    SubjectCreate,
    SubjectResponse,
)
from app.services.academic_service import AcademicService

router = APIRouter(prefix="/academic", tags=["Academic Structure"])


# ----------------- Academic Years -----------------
@router.post("/years", status_code=status.HTTP_201_CREATED, summary="Create Academic Year")
async def create_academic_year(
    payload: AcademicYearCreate,
    db: Session = Depends(get_db),
    admin_user: User = Depends(require_admin),
) -> Dict[str, Any]:
    year = AcademicService.create_academic_year(db, payload)
    return {
        "success": True,
        "message": "Academic year created successfully",
        "data": AcademicYearResponse.model_validate(year).model_dump(),
    }


@router.get("/years", summary="List Academic Years")
async def list_academic_years(
    active_only: bool = Query(False),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    years = AcademicService.list_academic_years(db, active_only)
    return {
        "success": True,
        "message": "Academic years retrieved",
        "data": [AcademicYearResponse.model_validate(y).model_dump() for y in years],
    }


# ----------------- Departments -----------------
@router.post("/departments", status_code=status.HTTP_201_CREATED, summary="Create Department")
async def create_department(
    payload: DepartmentCreate,
    db: Session = Depends(get_db),
    admin_user: User = Depends(require_admin),
) -> Dict[str, Any]:
    dept = AcademicService.create_department(db, payload)
    return {
        "success": True,
        "message": "Department created successfully",
        "data": DepartmentResponse.model_validate(dept).model_dump(),
    }


@router.get("/departments", summary="List Departments")
async def list_departments(
    academic_year_id: Optional[int] = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    depts = AcademicService.list_departments(db, academic_year_id)
    return {
        "success": True,
        "message": "Departments retrieved",
        "data": [DepartmentResponse.model_validate(d).model_dump() for d in depts],
    }


# ----------------- Programmes -----------------
@router.post("/programmes", status_code=status.HTTP_201_CREATED, summary="Create Programme")
async def create_programme(
    payload: ProgrammeCreate,
    db: Session = Depends(get_db),
    admin_user: User = Depends(require_admin),
) -> Dict[str, Any]:
    prog = AcademicService.create_programme(db, payload)
    return {
        "success": True,
        "message": "Programme created successfully",
        "data": ProgrammeResponse.model_validate(prog).model_dump(),
    }


@router.get("/programmes", summary="List Programmes")
async def list_programmes(
    department_id: Optional[int] = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    progs = AcademicService.list_programmes(db, department_id)
    return {
        "success": True,
        "message": "Programmes retrieved",
        "data": [ProgrammeResponse.model_validate(p).model_dump() for p in progs],
    }


# ----------------- Semesters -----------------
@router.post("/semesters", status_code=status.HTTP_201_CREATED, summary="Create Semester")
async def create_semester(
    payload: SemesterCreate,
    db: Session = Depends(get_db),
    admin_user: User = Depends(require_admin),
) -> Dict[str, Any]:
    sem = AcademicService.create_semester(db, payload)
    return {
        "success": True,
        "message": "Semester created successfully",
        "data": SemesterResponse.model_validate(sem).model_dump(),
    }


@router.get("/semesters", summary="List Semesters")
async def list_semesters(
    programme_id: Optional[int] = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    sems = AcademicService.list_semesters(db, programme_id)
    return {
        "success": True,
        "message": "Semesters retrieved",
        "data": [SemesterResponse.model_validate(s).model_dump() for s in sems],
    }


# ----------------- Class Sections -----------------
@router.post("/classes", status_code=status.HTTP_201_CREATED, summary="Create Class Section")
async def create_class_section(
    payload: ClassSectionCreate,
    db: Session = Depends(get_db),
    admin_user: User = Depends(require_admin),
) -> Dict[str, Any]:
    sec = AcademicService.create_class_section(db, payload)
    return {
        "success": True,
        "message": "Class section created successfully",
        "data": ClassSectionResponse.model_validate(sec).model_dump(),
    }


@router.get("/classes", summary="List Class Sections")
async def list_class_sections(
    semester_id: Optional[int] = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    classes = AcademicService.list_class_sections(db, semester_id)
    return {
        "success": True,
        "message": "Class sections retrieved",
        "data": [ClassSectionResponse.model_validate(c).model_dump() for c in classes],
    }


# ----------------- Subjects -----------------
@router.post("/subjects", status_code=status.HTTP_201_CREATED, summary="Create Subject")
async def create_subject(
    payload: SubjectCreate,
    db: Session = Depends(get_db),
    admin_user: User = Depends(require_admin),
) -> Dict[str, Any]:
    subject = AcademicService.create_subject(db, payload)
    return {
        "success": True,
        "message": "Subject created successfully",
        "data": SubjectResponse.model_validate(subject).model_dump(),
    }


@router.get("/subjects", summary="List Subjects")
async def list_subjects(
    class_section_id: Optional[int] = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    subjects = AcademicService.list_subjects(db, class_section_id)
    return {
        "success": True,
        "message": "Subjects retrieved",
        "data": [SubjectResponse.model_validate(s).model_dump() for s in subjects],
    }


# ----------------- Periods -----------------
@router.post("/periods", status_code=status.HTTP_201_CREATED, summary="Create Period")
async def create_period(
    payload: PeriodCreate,
    db: Session = Depends(get_db),
    admin_user: User = Depends(require_admin),
) -> Dict[str, Any]:
    period = AcademicService.create_period(db, payload)
    return {
        "success": True,
        "message": "Timetable period created successfully",
        "data": PeriodResponse.model_validate(period).model_dump(),
    }


@router.get("/periods", summary="List Periods")
async def list_periods(
    class_section_id: Optional[int] = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    periods = AcademicService.list_periods(db, class_section_id)
    return {
        "success": True,
        "message": "Periods retrieved",
        "data": [PeriodResponse.model_validate(p).model_dump() for p in periods],
    }


# ----------------- Faculty Assignments (BR-001) -----------------
@router.post("/assignments", status_code=status.HTTP_201_CREATED, summary="Assign Faculty to Class & Subject")
async def assign_faculty(
    payload: FacultyAssignmentCreate,
    db: Session = Depends(get_db),
    admin_user: User = Depends(require_admin),
) -> Dict[str, Any]:
    """Admin assigns a faculty member to teach a subject in a specific class."""
    assignment = AcademicService.assign_faculty(
        db=db,
        data=payload,
        assigned_by_admin_id=admin_user.id,
    )
    return {
        "success": True,
        "message": "Faculty assigned to class and subject successfully",
        "data": FacultyAssignmentResponse.model_validate(assignment).model_dump(),
    }


@router.get("/faculty/my-classes", summary="Get Current Faculty's Assigned Classes")
async def get_my_assigned_classes(
    db: Session = Depends(get_db),
    faculty_user: User = Depends(require_faculty),
) -> Dict[str, Any]:
    """
    Returns only the classes & subjects assigned to the logged-in faculty member (BR-001).
    Used directly by the Faculty Dashboard attendance dropdown.
    """
    assigned = AcademicService.list_faculty_assigned_classes(db, faculty_user.id)
    return {
        "success": True,
        "message": "Assigned classes retrieved",
        "data": [a.model_dump() for a in assigned],
    }

