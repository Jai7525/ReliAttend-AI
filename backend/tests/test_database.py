"""
ReliAttend AI — Database Models & Integrity Verification Tests
Tests table creation, schema integrity, and duplicate prevention constraints.
"""

from datetime import date, datetime, time, timezone
import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models import (
    User,
    AcademicYear,
    Department,
    Programme,
    Semester,
    ClassSection,
    Subject,
    Period,
    FacultyAssignment,
    Student,
    ConsentRecord,
    FaceReference,
    ModelConfig,
    AttendanceSession,
    DetectionEvidence,
    AttendanceRecord,
    ReviewDecision,
    AttendanceCorrection,
    AuditLog,
)

# In-memory SQLite engine for fast, isolated test execution
TEST_DB_URL = "sqlite:///:memory:"


@pytest.fixture(scope="function")
def db_session():
    """Provides a fresh isolated database schema and session for each test."""
    engine = create_engine(TEST_DB_URL, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


def test_all_expected_tables_registered():
    """Verify that all 19 domain tables are registered with Base metadata."""
    expected_tables = {
        "users",
        "academic_years",
        "departments",
        "programmes",
        "semesters",
        "class_sections",
        "subjects",
        "periods",
        "faculty_assignments",
        "students",
        "consent_records",
        "face_references",
        "model_configs",
        "attendance_sessions",
        "detection_evidence",
        "attendance_records",
        "review_decisions",
        "attendance_corrections",
        "audit_logs",
    }
    actual_tables = set(Base.metadata.tables.keys())
    for table in expected_tables:
        assert table in actual_tables, f"Table '{table}' missing from metadata!"


def test_user_creation_and_query(db_session):
    """Test user model insertion and query."""
    user = User(
        email="faculty@reliattend.com",
        username="prof_sharma",
        password_hash="$2b$12$fakehashedpasswordfortesting123",
        full_name="Prof. Sharma",
        role="faculty",
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)

    assert user.id is not None
    assert user.role == "faculty"
    assert user.is_active is True


def test_academic_hierarchy_creation(db_session):
    """Test full academic structure hierarchy insertion."""
    year = AcademicYear(
        name="2026-2027",
        start_date=date(2026, 8, 1),
        end_date=date(2027, 5, 31),
    )
    db_session.add(year)
    db_session.commit()

    dept = Department(
        name="Computer Science & Engineering",
        code="CSE",
        academic_year_id=year.id,
    )
    db_session.add(dept)
    db_session.commit()

    prog = Programme(
        name="B.Tech Artificial Intelligence",
        code="BTECH-AI",
        department_id=dept.id,
        duration_years=4,
    )
    db_session.add(prog)
    db_session.commit()

    sem = Semester(
        name="Semester 5",
        number=5,
        programme_id=prog.id,
    )
    db_session.add(sem)
    db_session.commit()

    sec = ClassSection(
        name="Section A",
        semester_id=sem.id,
        max_students=60,
    )
    db_session.add(sec)
    db_session.commit()

    subject = Subject(
        name="Computer Vision",
        code="CS501",
        class_section_id=sec.id,
    )
    db_session.add(subject)
    db_session.commit()

    period = Period(
        name="Period 1",
        day_of_week="Monday",
        start_time=time(9, 0),
        end_time=time(9, 50),
        subject_id=subject.id,
        class_section_id=sec.id,
    )
    db_session.add(period)
    db_session.commit()

    assert period.id is not None
    assert subject.class_section.name == "Section A"


def test_student_and_biometric_face_reference(db_session):
    """Test Student creation with ConsentRecord and FaceReference binary embedding."""
    # Setup prerequisite academic section
    year = AcademicYear(name="2026-2027", start_date=date(2026, 8, 1), end_date=date(2027, 5, 31))
    db_session.add(year)
    db_session.commit()
    dept = Department(name="CSE", code="CSE", academic_year_id=year.id)
    db_session.add(dept)
    db_session.commit()
    prog = Programme(name="BTech", code="BT", department_id=dept.id, duration_years=4)
    db_session.add(prog)
    db_session.commit()
    sem = Semester(name="Sem 1", number=1, programme_id=prog.id)
    db_session.add(sem)
    db_session.commit()
    sec = ClassSection(name="Sec A", semester_id=sem.id)
    db_session.add(sec)
    db_session.commit()

    # Create Student
    student = Student(
        student_id="STU2026001",
        name="Aarav Patel",
        roll_number="26CS001",
        class_section_id=sec.id,
        registration_status="enrolled",
    )
    db_session.add(student)
    db_session.commit()

    # Add Biometric Consent (BR-018)
    consent = ConsentRecord(
        student_id=student.id,
        consent_type="institutional_enrollment",
        consent_reference="FORM-2026-001",
    )
    db_session.add(consent)
    db_session.commit()

    # Add Face Reference with 512-byte dummy binary embedding
    dummy_embedding = b"\x00" * 512
    face_ref = FaceReference(
        student_id=student.id,
        image_path="/storage/reference_images/STU2026001.jpg",
        embedding=dummy_embedding,
        embedding_size=512,
        model_name="insightface_buffalo_l",
        model_version="1.0",
        quality_score=0.92,
    )
    db_session.add(face_ref)
    db_session.commit()

    # Query verification
    retrieved = db_session.query(Student).filter_by(student_id="STU2026001").first()
    assert retrieved is not None
    assert retrieved.consent_record.consent_reference == "FORM-2026-001"
    assert retrieved.face_reference.embedding == dummy_embedding
    assert retrieved.face_reference.model_name == "insightface_buffalo_l"


def test_duplicate_attendance_prevention_constraint(db_session):
    """
    CRITICAL (BR-012, FR-ATT-001):
    Verify that the database UNIQUE constraint prevents duplicate attendance records
    for the same student in the same session.
    """
    # Setup minimum session and student prerequisites
    year = AcademicYear(name="2026-2027", start_date=date(2026, 8, 1), end_date=date(2027, 5, 31))
    dept = Department(name="CSE", code="CSE", academic_year=year)
    prog = Programme(name="BTech", code="BT", department=dept)
    sem = Semester(name="Sem 1", number=1, programme=prog)
    sec = ClassSection(name="Sec A", semester=sem)
    subject = Subject(name="CV", code="CV101", class_section=sec)
    period = Period(name="P1", day_of_week="Mon", start_time=time(9, 0), end_time=time(9, 50), subject=subject, class_section=sec)
    faculty = User(email="f@test.com", username="f", password_hash="hash", full_name="F", role="faculty")
    student = Student(student_id="STU1", name="Alice", roll_number="1", class_section=sec)

    db_session.add_all([year, dept, prog, sem, sec, subject, period, faculty, student])
    db_session.commit()

    session = AttendanceSession(
        faculty_id=faculty.id,
        class_section_id=sec.id,
        subject_id=subject.id,
        period_id=period.id,
        session_date=date(2026, 11, 4),
        start_time=datetime.now(timezone.utc),
    )
    db_session.add(session)
    db_session.commit()

    # First attendance record: should succeed
    record1 = AttendanceRecord(
        session_id=session.id,
        student_id=student.id,
        status="confirmed",
        source="auto",
        reliability_score=0.91,
    )
    db_session.add(record1)
    db_session.commit()

    # Second attendance record for the SAME session and student: MUST FAIL WITH INTEGRITY ERROR
    record2 = AttendanceRecord(
        session_id=session.id,
        student_id=student.id,
        status="confirmed",
        source="auto",
        reliability_score=0.88,
    )
    db_session.add(record2)

    with pytest.raises(IntegrityError):
        db_session.commit()

    db_session.rollback()


def test_duplicate_faculty_assignment_prevention(db_session):
    """
    Verify BR-001 constraint: Faculty cannot be assigned twice to the exact same class and subject.
    """
    year = AcademicYear(name="2026-2027", start_date=date(2026, 8, 1), end_date=date(2027, 5, 31))
    dept = Department(name="CSE", code="CSE", academic_year=year)
    prog = Programme(name="BTech", code="BT", department=dept)
    sem = Semester(name="Sem 1", number=1, programme=prog)
    sec = ClassSection(name="Sec A", semester=sem)
    subject = Subject(name="Math", code="M101", class_section=sec)
    admin = User(email="a@test.com", username="a", password_hash="h", full_name="A", role="admin")
    faculty = User(email="f@test.com", username="f", password_hash="h", full_name="F", role="faculty")

    db_session.add_all([year, dept, prog, sem, sec, subject, admin, faculty])
    db_session.commit()

    assign1 = FacultyAssignment(
        user_id=faculty.id,
        class_section_id=sec.id,
        subject_id=subject.id,
        assigned_by=admin.id,
    )
    db_session.add(assign1)
    db_session.commit()

    # Duplicate assignment
    assign2 = FacultyAssignment(
        user_id=faculty.id,
        class_section_id=sec.id,
        subject_id=subject.id,
        assigned_by=admin.id,
    )
    db_session.add(assign2)

    with pytest.raises(IntegrityError):
        db_session.commit()

    db_session.rollback()

