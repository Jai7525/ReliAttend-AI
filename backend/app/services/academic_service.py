"""
ReliAttend AI — Academic Hierarchy Business Logic Service
Handles operations on academic years, departments, programmes, semesters,
classes, subjects, timetable periods, and faculty assignments.
"""

from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from app.core.exceptions import (
    DuplicateResourceError,
    ResourceNotFoundError,
    ValidationProcessingError,
)
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
from app.models.user import User
from app.schemas.academic import (
    AcademicYearCreate,
    AcademicYearUpdate,
    DepartmentCreate,
    DepartmentUpdate,
    ProgrammeCreate,
    SemesterCreate,
    ClassSectionCreate,
    ClassSectionUpdate,
    SubjectCreate,
    PeriodCreate,
    FacultyAssignmentCreate,
    AssignedClassDetail,
)


class AcademicService:
    """Service handling academic structure and timetable management."""

    # ----------------- Academic Years -----------------
    @staticmethod
    def create_academic_year(db: Session, data: AcademicYearCreate) -> AcademicYear:
        existing = db.query(AcademicYear).filter_by(name=data.name).first()
        if existing:
            raise DuplicateResourceError(f"Academic year '{data.name}' already exists.")
        if data.start_date >= data.end_date:
            raise ValidationProcessingError("Academic year start_date must precede end_date.")

        year = AcademicYear(**data.model_dump())
        db.add(year)
        db.commit()
        db.refresh(year)
        return year

    @staticmethod
    def list_academic_years(db: Session, active_only: bool = False) -> List[AcademicYear]:
        query = db.query(AcademicYear)
        if active_only:
            query = query.filter_by(is_active=True)
        return query.order_by(AcademicYear.start_date.desc()).all()

    @staticmethod
    def get_academic_year(db: Session, year_id: int) -> AcademicYear:
        year = db.query(AcademicYear).filter_by(id=year_id).first()
        if not year:
            raise ResourceNotFoundError(f"Academic year ID {year_id} not found.")
        return year

    @staticmethod
    def update_academic_year(db: Session, year_id: int, data: AcademicYearUpdate) -> AcademicYear:
        year = AcademicService.get_academic_year(db, year_id)
        update_dict = data.model_dump(exclude_unset=True)
        for key, value in update_dict.items():
            setattr(year, key, value)
        db.commit()
        db.refresh(year)
        return year

    # ----------------- Departments -----------------
    @staticmethod
    def create_department(db: Session, data: DepartmentCreate) -> Department:
        AcademicService.get_academic_year(db, data.academic_year_id)
        existing = db.query(Department).filter_by(
            code=data.code.upper(),
            academic_year_id=data.academic_year_id,
        ).first()
        if existing:
            raise DuplicateResourceError(f"Department code '{data.code}' already exists for this academic year.")

        dept = Department(
            name=data.name,
            code=data.code.upper(),
            academic_year_id=data.academic_year_id,
            is_active=data.is_active,
        )
        db.add(dept)
        db.commit()
        db.refresh(dept)
        return dept

    @staticmethod
    def list_departments(db: Session, academic_year_id: Optional[int] = None) -> List[Department]:
        query = db.query(Department)
        if academic_year_id:
            query = query.filter_by(academic_year_id=academic_year_id)
        return query.order_by(Department.name.asc()).all()

    # ----------------- Programmes -----------------
    @staticmethod
    def create_programme(db: Session, data: ProgrammeCreate) -> Programme:
        dept = db.query(Department).filter_by(id=data.department_id).first()
        if not dept:
            raise ResourceNotFoundError(f"Department ID {data.department_id} not found.")

        existing = db.query(Programme).filter_by(
            code=data.code.upper(),
            department_id=data.department_id,
        ).first()
        if existing:
            raise DuplicateResourceError(f"Programme code '{data.code}' already exists in this department.")

        prog = Programme(
            name=data.name,
            code=data.code.upper(),
            department_id=data.department_id,
            duration_years=data.duration_years,
            is_active=data.is_active,
        )
        db.add(prog)
        db.commit()
        db.refresh(prog)
        return prog

    @staticmethod
    def list_programmes(db: Session, department_id: Optional[int] = None) -> List[Programme]:
        query = db.query(Programme)
        if department_id:
            query = query.filter_by(department_id=department_id)
        return query.all()

    # ----------------- Semesters -----------------
    @staticmethod
    def create_semester(db: Session, data: SemesterCreate) -> Semester:
        prog = db.query(Programme).filter_by(id=data.programme_id).first()
        if not prog:
            raise ResourceNotFoundError(f"Programme ID {data.programme_id} not found.")

        existing = db.query(Semester).filter_by(
            number=data.number,
            programme_id=data.programme_id,
        ).first()
        if existing:
            raise DuplicateResourceError(f"Semester {data.number} already exists for this programme.")

        sem = Semester(**data.model_dump())
        db.add(sem)
        db.commit()
        db.refresh(sem)
        return sem

    @staticmethod
    def list_semesters(db: Session, programme_id: Optional[int] = None) -> List[Semester]:
        query = db.query(Semester)
        if programme_id:
            query = query.filter_by(programme_id=programme_id)
        return query.order_by(Semester.number.asc()).all()

    # ----------------- Class Sections -----------------
    @staticmethod
    def create_class_section(db: Session, data: ClassSectionCreate) -> ClassSection:
        sem = db.query(Semester).filter_by(id=data.semester_id).first()
        if not sem:
            raise ResourceNotFoundError(f"Semester ID {data.semester_id} not found.")

        existing = db.query(ClassSection).filter_by(
            name=data.name,
            semester_id=data.semester_id,
        ).first()
        if existing:
            raise DuplicateResourceError(f"Class section '{data.name}' already exists in this semester.")

        sec = ClassSection(**data.model_dump())
        db.add(sec)
        db.commit()
        db.refresh(sec)
        return sec

    @staticmethod
    def list_class_sections(db: Session, semester_id: Optional[int] = None) -> List[ClassSection]:
        query = db.query(ClassSection)
        if semester_id:
            query = query.filter_by(semester_id=semester_id)
        return query.all()

    @staticmethod
    def get_class_section(db: Session, class_id: int) -> ClassSection:
        sec = db.query(ClassSection).filter_by(id=class_id).first()
        if not sec:
            raise ResourceNotFoundError(f"Class section ID {class_id} not found.")
        return sec

    # ----------------- Subjects -----------------
    @staticmethod
    def create_subject(db: Session, data: SubjectCreate) -> Subject:
        sec = db.query(ClassSection).filter_by(id=data.class_section_id).first()
        if not sec:
            raise ResourceNotFoundError(f"Class section ID {data.class_section_id} not found.")

        existing = db.query(Subject).filter_by(
            code=data.code.upper(),
            class_section_id=data.class_section_id,
        ).first()
        if existing:
            raise DuplicateResourceError(f"Subject code '{data.code}' already exists for this class.")

        sub = Subject(
            name=data.name,
            code=data.code.upper(),
            class_section_id=data.class_section_id,
            is_active=data.is_active,
        )
        db.add(sub)
        db.commit()
        db.refresh(sub)
        return sub

    @staticmethod
    def list_subjects(db: Session, class_section_id: Optional[int] = None) -> List[Subject]:
        query = db.query(Subject)
        if class_section_id:
            query = query.filter_by(class_section_id=class_section_id)
        return query.all()

    # ----------------- Periods -----------------
    @staticmethod
    def create_period(db: Session, data: PeriodCreate) -> Period:
        sub = db.query(Subject).filter_by(id=data.subject_id).first()
        if not sub:
            raise ResourceNotFoundError(f"Subject ID {data.subject_id} not found.")
        sec = db.query(ClassSection).filter_by(id=data.class_section_id).first()
        if not sec:
            raise ResourceNotFoundError(f"Class section ID {data.class_section_id} not found.")

        period = Period(**data.model_dump())
        db.add(period)
        db.commit()
        db.refresh(period)
        return period

    @staticmethod
    def list_periods(db: Session, class_section_id: Optional[int] = None) -> List[Period]:
        query = db.query(Period)
        if class_section_id:
            query = query.filter_by(class_section_id=class_section_id)
        return query.all()

    # ----------------- Faculty Assignments (BR-001) -----------------
    @staticmethod
    def assign_faculty(
        db: Session,
        data: FacultyAssignmentCreate,
        assigned_by_admin_id: int,
    ) -> FacultyAssignment:
        """Assign faculty to class & subject, enforcing BR-001."""
        faculty_user = db.query(User).filter_by(id=data.user_id, role="faculty").first()
        if not faculty_user:
            raise ResourceNotFoundError(f"Faculty user ID {data.user_id} not found.")

        sec = db.query(ClassSection).filter_by(id=data.class_section_id).first()
        if not sec:
            raise ResourceNotFoundError(f"Class section ID {data.class_section_id} not found.")

        sub = db.query(Subject).filter_by(id=data.subject_id).first()
        if not sub:
            raise ResourceNotFoundError(f"Subject ID {data.subject_id} not found.")

        existing = db.query(FacultyAssignment).filter_by(
            user_id=data.user_id,
            class_section_id=data.class_section_id,
            subject_id=data.subject_id,
        ).first()
        if existing:
            raise DuplicateResourceError("Faculty is already assigned to this class and subject.")

        assignment = FacultyAssignment(
            user_id=data.user_id,
            class_section_id=data.class_section_id,
            subject_id=data.subject_id,
            is_active=data.is_active,
            assigned_by=assigned_by_admin_id,
        )
        db.add(assignment)
        db.commit()
        db.refresh(assignment)
        return assignment

    @staticmethod
    def list_faculty_assigned_classes(db: Session, user_id: int) -> List[AssignedClassDetail]:
        """Retrieve assigned classes and subjects for faculty dashboard."""
        assignments = db.query(FacultyAssignment).filter_by(user_id=user_id, is_active=True).all()
        result = []
        for a in assignments:
            result.append(
                AssignedClassDetail(
                    assignment_id=a.id,
                    class_section_id=a.class_section_id,
                    class_section_name=a.class_section.name,
                    subject_id=a.subject_id,
                    subject_name=a.subject.name,
                    subject_code=a.subject.code,
                )
            )
        return result

