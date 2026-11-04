"""
ReliAttend AI — Student Management Endpoints Router
Exposes APIs for student profiles, class enrollment, and biometric consent logging.
"""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.user import User
from app.routes.deps import get_current_user, require_admin
from app.schemas.student import (
    ConsentRecordCreate,
    ConsentRecordResponse,
    StudentCreate,
    StudentResponse,
    StudentUpdate,
)
from app.services.student_service import StudentService

router = APIRouter(prefix="/students", tags=["Student Management"])


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    summary="Register New Student",
    description="Create student profile with unique student ID and class assignment.",
)
async def create_student(
    payload: StudentCreate,
    db: Session = Depends(get_db),
    admin_user: User = Depends(require_admin),
) -> Dict[str, Any]:
    student = StudentService.create_student(db, payload)
    return {
        "success": True,
        "message": "Student registered successfully",
        "data": StudentService.build_student_response(student).model_dump(),
    }


@router.get(
    "",
    summary="List Students",
    description="Retrieve students with optional filtering by class section and active status.",
)
async def list_students(
    class_section_id: Optional[int] = Query(None, description="Filter by class section ID"),
    active_only: bool = Query(True, description="Filter active students only"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    students = StudentService.list_students(db, class_section_id, active_only)
    return {
        "success": True,
        "message": "Students retrieved successfully",
        "data": [StudentService.build_student_response(s).model_dump() for s in students],
    }


@router.get(
    "/{student_id}",
    summary="Get Student Details",
    description="Retrieve single student profile, enrollment status, and consent status.",
)
async def get_student(
    student_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    student = StudentService.get_student(db, student_id)
    return {
        "success": True,
        "message": "Student details retrieved successfully",
        "data": StudentService.build_student_response(student).model_dump(),
    }


@router.put(
    "/{student_id}",
    summary="Update Student Profile",
    description="Update student details such as name, roll number, or assigned class.",
)
async def update_student(
    student_id: int,
    payload: StudentUpdate,
    db: Session = Depends(get_db),
    admin_user: User = Depends(require_admin),
) -> Dict[str, Any]:
    student = StudentService.update_student(db, student_id, payload)
    return {
        "success": True,
        "message": "Student updated successfully",
        "data": StudentService.build_student_response(student).model_dump(),
    }


@router.patch(
    "/{student_id}/deactivate",
    summary="Deactivate Student",
    description="Deactivate a student to exclude them from active recognition scopes.",
)
async def deactivate_student(
    student_id: int,
    db: Session = Depends(get_db),
    admin_user: User = Depends(require_admin),
) -> Dict[str, Any]:
    student = StudentService.deactivate_student(db, student_id)
    return {
        "success": True,
        "message": "Student deactivated successfully",
        "data": StudentService.build_student_response(student).model_dump(),
    }


@router.post(
    "/{student_id}/consent",
    status_code=status.HTTP_201_CREATED,
    summary="Record Biometric Consent",
    description="Log institutional authorization/consent metadata for student biometric processing (BR-018, FR-ADMIN-004).",
)
async def record_biometric_consent(
    student_id: int,
    payload: ConsentRecordCreate,
    db: Session = Depends(get_db),
    admin_user: User = Depends(require_admin),
) -> Dict[str, Any]:
    consent = StudentService.record_consent(
        db=db,
        student_id=student_id,
        data=payload,
        granted_by_user_id=admin_user.id,
    )
    return {
        "success": True,
        "message": "Biometric consent recorded successfully",
        "data": ConsentRecordResponse.model_validate(consent).model_dump(),
    }

