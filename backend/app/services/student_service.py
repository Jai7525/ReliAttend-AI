"""
ReliAttend AI — Student Business Logic Service
Handles student creation, validation, unique constraints, and biometric consent logging.
"""

from typing import List, Optional
from sqlalchemy.orm import Session

from app.core.exceptions import (
    DuplicateResourceError,
    ResourceNotFoundError,
)
from app.models.academic import ClassSection
from app.models.student import ConsentRecord, Student
from app.schemas.student import (
    ConsentRecordCreate,
    StudentCreate,
    StudentResponse,
    StudentUpdate,
)


class StudentService:
    """Service handling student registration, profiles, and consent metadata."""

    @staticmethod
    def create_student(db: Session, data: StudentCreate) -> Student:
        """Create new student record with uniqueness verification."""
        # 1. Verify that class section exists
        sec = db.query(ClassSection).filter_by(id=data.class_section_id).first()
        if not sec:
            raise ResourceNotFoundError(f"Class section ID {data.class_section_id} not found.")

        # 2. Check for duplicate institutional student ID
        existing_student_id = db.query(Student).filter_by(student_id=data.student_id).first()
        if existing_student_id:
            raise DuplicateResourceError(f"Student ID '{data.student_id}' is already registered.")

        # 3. Check for duplicate roll number within the same class section
        existing_roll = db.query(Student).filter_by(
            roll_number=data.roll_number,
            class_section_id=data.class_section_id,
        ).first()
        if existing_roll:
            raise DuplicateResourceError(
                f"Roll number '{data.roll_number}' already exists in class section '{sec.name}'."
            )

        student = Student(
            student_id=data.student_id,
            name=data.name,
            roll_number=data.roll_number,
            class_section_id=data.class_section_id,
            registration_status="pending",
            is_active=True,
        )
        db.add(student)
        db.commit()
        db.refresh(student)
        return student

    @staticmethod
    def list_students(
        db: Session,
        class_section_id: Optional[int] = None,
        active_only: bool = True,
    ) -> List[Student]:
        """Query students with optional class filter."""
        query = db.query(Student)
        if class_section_id is not None:
            query = query.filter_by(class_section_id=class_section_id)
        if active_only:
            query = query.filter_by(is_active=True)
        return query.order_by(Student.roll_number.asc()).all()

    @staticmethod
    def get_student(db: Session, student_id: int) -> Student:
        """Retrieve student by primary key ID."""
        student = db.query(Student).filter_by(id=student_id).first()
        if not student:
            raise ResourceNotFoundError(f"Student ID {student_id} not found.")
        return student

    @staticmethod
    def update_student(db: Session, student_id: int, data: StudentUpdate) -> Student:
        """Update student details."""
        student = StudentService.get_student(db, student_id)
        update_dict = data.model_dump(exclude_unset=True)

        if "class_section_id" in update_dict:
            sec = db.query(ClassSection).filter_by(id=update_dict["class_section_id"]).first()
            if not sec:
                raise ResourceNotFoundError(f"Class section ID {update_dict['class_section_id']} not found.")

        for key, value in update_dict.items():
            setattr(student, key, value)

        db.commit()
        db.refresh(student)
        return student

    @staticmethod
    def deactivate_student(db: Session, student_id: int) -> Student:
        """Deactivate student, removing them from live recognition scope."""
        student = StudentService.get_student(db, student_id)
        student.is_active = False
        student.registration_status = "deactivated"
        db.commit()
        db.refresh(student)
        return student

    @staticmethod
    def record_consent(
        db: Session,
        student_id: int,
        data: ConsentRecordCreate,
        granted_by_user_id: int,
    ) -> ConsentRecord:
        """
        Record biometric authorization/consent metadata (BR-018, FR-ADMIN-004).
        Prerequisite for face reference image and embedding enrolment.
        """
        student = StudentService.get_student(db, student_id)

        existing_consent = db.query(ConsentRecord).filter_by(student_id=student.id).first()
        if existing_consent:
            existing_consent.consent_type = data.consent_type
            existing_consent.consent_reference = data.consent_reference
            existing_consent.granted_by = granted_by_user_id
            existing_consent.is_active = True
            db.commit()
            db.refresh(existing_consent)
            return existing_consent

        consent = ConsentRecord(
            student_id=student.id,
            consent_type=data.consent_type,
            consent_reference=data.consent_reference,
            granted_by=granted_by_user_id,
            is_active=True,
        )
        db.add(consent)
        db.commit()
        db.refresh(consent)
        return consent

    @staticmethod
    def build_student_response(student: Student) -> StudentResponse:
        """Helper to serialize student with consent and biometric flags."""
        has_consent = (
            student.consent_record is not None and student.consent_record.is_active
        )
        has_face_data = (
            student.face_reference is not None and student.face_reference.status == "active"
        )
        return StudentResponse(
            id=student.id,
            student_id=student.student_id,
            name=student.name,
            roll_number=student.roll_number,
            class_section_id=student.class_section_id,
            registration_status=student.registration_status,
            is_active=student.is_active,
            created_at=student.created_at,
            has_consent=has_consent,
            has_face_data=has_face_data,
        )

