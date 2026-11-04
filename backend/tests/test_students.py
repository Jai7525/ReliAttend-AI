"""
ReliAttend AI — Student Management & Consent Automated Tests
Tests student CRUD, duplicate ID and roll number prevention, RBAC, and consent logging.
"""

from datetime import date
import uuid
from fastapi import status
from fastapi.testclient import TestClient

from app.core.security import hash_password
from app.database import Base, SessionLocal, engine
from app.main import app
from app.models.academic import AcademicYear, ClassSection, Department, Programme, Semester
from app.models.user import User

client = TestClient(app)

ADMIN_TOKEN = ""
FACULTY_TOKEN = ""
CLASS_SECTION_ID_1 = 0
CLASS_SECTION_ID_2 = 0
TEST_UID = uuid.uuid4().hex[:6].upper()


def setup_module():
    """Ensure prerequisite users and academic class sections exist."""
    global ADMIN_TOKEN, FACULTY_TOKEN, CLASS_SECTION_ID_1, CLASS_SECTION_ID_2
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        # Create users
        admin = db.query(User).filter_by(username="stu_admin").first()
        if not admin:
            admin = User(
                email="stu_admin@reliattend.com",
                username="stu_admin",
                password_hash=hash_password("Pass123!"),
                full_name="Student Module Admin",
                role="admin",
                is_active=True,
            )
            db.add(admin)

        faculty = db.query(User).filter_by(username="stu_faculty").first()
        if not faculty:
            faculty = User(
                email="stu_faculty@reliattend.com",
                username="stu_faculty",
                password_hash=hash_password("Pass123!"),
                full_name="Student Module Faculty",
                role="faculty",
                is_active=True,
            )
            db.add(faculty)

        # Create academic hierarchy
        year = AcademicYear(name=f"2026-STU-{TEST_UID}", start_date=date(2026, 8, 1), end_date=date(2027, 5, 31))
        db.add(year)
        db.commit()

        dept = Department(name="CSE-Stu", code=f"S{TEST_UID[:3]}", academic_year_id=year.id)
        db.add(dept)
        db.commit()

        prog = Programme(name="BTech-Stu", code=f"B{TEST_UID[:3]}", department_id=dept.id, duration_years=4)
        db.add(prog)
        db.commit()

        sem = Semester(name="Sem 1", number=1, programme_id=prog.id)
        db.add(sem)
        db.commit()

        sec1 = ClassSection(name=f"Sec Alpha {TEST_UID}", semester_id=sem.id, max_students=60)
        sec2 = ClassSection(name=f"Sec Beta {TEST_UID}", semester_id=sem.id, max_students=60)
        db.add_all([sec1, sec2])
        db.commit()

        CLASS_SECTION_ID_1 = sec1.id
        CLASS_SECTION_ID_2 = sec2.id
    finally:
        db.close()

    # Obtain tokens
    res_admin = client.post(
        "/api/v1/auth/login",
        json={"username_or_email": "stu_admin", "password": "Pass123!"},
    )
    ADMIN_TOKEN = res_admin.json()["data"]["tokens"]["access_token"]

    res_faculty = client.post(
        "/api/v1/auth/login",
        json={"username_or_email": "stu_faculty", "password": "Pass123!"},
    )
    FACULTY_TOKEN = res_faculty.json()["data"]["tokens"]["access_token"]


def test_register_student_admin_success():
    """Verify Admin can register a student."""
    response = client.post(
        "/api/v1/students",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        json={
            "student_id": f"STU_{TEST_UID}_001",
            "name": "Aarav Patel",
            "roll_number": "26CS001",
            "class_section_id": CLASS_SECTION_ID_1,
        },
    )
    assert response.status_code == status.HTTP_201_CREATED
    data = response.json()["data"]
    assert data["student_id"] == f"STU_{TEST_UID}_001"
    assert data["name"] == "Aarav Patel"
    assert data["has_consent"] is False


def test_duplicate_student_id_rejected():
    """Verify duplicate institutional student ID is rejected (FR-STU-002)."""
    response = client.post(
        "/api/v1/students",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        json={
            "student_id": f"STU_{TEST_UID}_001",
            "name": "Different Name",
            "roll_number": "26CS999",
            "class_section_id": CLASS_SECTION_ID_1,
        },
    )
    assert response.status_code == status.HTTP_409_CONFLICT


def test_duplicate_roll_number_in_same_class_rejected():
    """Verify duplicate roll number within the same class section is rejected."""
    response = client.post(
        "/api/v1/students",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        json={
            "student_id": f"STU_{TEST_UID}_002",
            "name": "Priya Sharma",
            "roll_number": "26CS001",  # Same roll number as Aarav in Section Alpha
            "class_section_id": CLASS_SECTION_ID_1,
        },
    )
    assert response.status_code == status.HTTP_409_CONFLICT


def test_same_roll_number_in_different_class_allowed():
    """Verify identical roll number is permitted in a different class section."""
    response = client.post(
        "/api/v1/students",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        json={
            "student_id": f"STU_{TEST_UID}_003",
            "name": "Rahul Verma",
            "roll_number": "26CS001",  # Same roll number, but in Section Beta
            "class_section_id": CLASS_SECTION_ID_2,
        },
    )
    assert response.status_code == status.HTTP_201_CREATED


def test_faculty_cannot_register_student_forbidden():
    """Verify Faculty role cannot register students (Admin RBAC)."""
    response = client.post(
        "/api/v1/students",
        headers={"Authorization": f"Bearer {FACULTY_TOKEN}"},
        json={
            "student_id": f"STU_{TEST_UID}_HACK",
            "name": "Hacker",
            "roll_number": "999",
            "class_section_id": CLASS_SECTION_ID_1,
        },
    )
    assert response.status_code == status.HTTP_403_FORBIDDEN


def test_record_biometric_consent_success():
    """Verify recording biometric consent metadata (BR-018, FR-ADMIN-004)."""
    list_res = client.get(
        "/api/v1/students",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        params={"class_section_id": CLASS_SECTION_ID_1},
    )
    student = [s for s in list_res.json()["data"] if s["student_id"] == f"STU_{TEST_UID}_001"][0]
    student_pk = student["id"]

    consent_res = client.post(
        f"/api/v1/students/{student_pk}/consent",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        json={
            "consent_type": "institutional_enrollment",
            "consent_reference": f"FORM-2026-{TEST_UID}",
        },
    )
    assert consent_res.status_code == status.HTTP_201_CREATED
    assert consent_res.json()["data"]["consent_reference"] == f"FORM-2026-{TEST_UID}"

    detail_res = client.get(
        f"/api/v1/students/{student_pk}",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    assert detail_res.status_code == status.HTTP_200_OK
    assert detail_res.json()["data"]["has_consent"] is True


def test_deactivate_student():
    """Verify student deactivation excludes them from active scopes."""
    list_res = client.get(
        "/api/v1/students",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        params={"class_section_id": CLASS_SECTION_ID_2},
    )
    student_pk = list_res.json()["data"][0]["id"]

    deact_res = client.patch(
        f"/api/v1/students/{student_pk}/deactivate",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    assert deact_res.status_code == status.HTTP_200_OK
    assert deact_res.json()["data"]["is_active"] is False
    assert deact_res.json()["data"]["registration_status"] == "deactivated"
