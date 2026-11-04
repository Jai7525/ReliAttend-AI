"""
ReliAttend AI — Academic Hierarchy & RBAC Automated Tests
Tests creation, retrieval, duplicate rejection, and faculty assignment enforcement.
"""

import uuid
from fastapi import status
from fastapi.testclient import TestClient

from app.core.security import hash_password
from app.database import Base, SessionLocal, engine
from app.main import app
from app.models.user import User

client = TestClient(app)

ADMIN_TOKEN = ""
FACULTY_TOKEN = ""
FACULTY_ID = 0
TEST_UID = uuid.uuid4().hex[:6].upper()


def setup_module():
    """Seed test admin and faculty accounts and obtain JWT tokens."""
    global ADMIN_TOKEN, FACULTY_TOKEN, FACULTY_ID
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        admin = db.query(User).filter_by(username="acad_admin").first()
        if not admin:
            admin = User(
                email="acad_admin@reliattend.com",
                username="acad_admin",
                password_hash=hash_password("Pass123!"),
                full_name="Academic Admin",
                role="admin",
                is_active=True,
            )
            db.add(admin)

        faculty = db.query(User).filter_by(username="acad_faculty").first()
        if not faculty:
            faculty = User(
                email="acad_faculty@reliattend.com",
                username="acad_faculty",
                password_hash=hash_password("Pass123!"),
                full_name="Academic Faculty",
                role="faculty",
                is_active=True,
            )
            db.add(faculty)

        db.commit()
        db.refresh(faculty)
        FACULTY_ID = faculty.id
    finally:
        db.close()

    # Obtain tokens
    res_admin = client.post(
        "/api/v1/auth/login",
        json={"username_or_email": "acad_admin", "password": "Pass123!"},
    )
    ADMIN_TOKEN = res_admin.json()["data"]["tokens"]["access_token"]

    res_faculty = client.post(
        "/api/v1/auth/login",
        json={"username_or_email": "acad_faculty", "password": "Pass123!"},
    )
    FACULTY_TOKEN = res_faculty.json()["data"]["tokens"]["access_token"]


def test_create_academic_year_admin_success():
    """Verify Admin can create an academic year."""
    year_name = f"2026-YEAR-{TEST_UID}"
    response = client.post(
        "/api/v1/academic/years",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        json={"name": year_name, "start_date": "2026-08-01", "end_date": "2027-05-31"},
    )
    assert response.status_code == status.HTTP_201_CREATED
    data = response.json()["data"]
    assert data["name"] == year_name


def test_create_academic_year_faculty_forbidden():
    """Verify Faculty cannot create an academic year (RBAC check)."""
    response = client.post(
        "/api/v1/academic/years",
        headers={"Authorization": f"Bearer {FACULTY_TOKEN}"},
        json={"name": f"2027-HACK-{TEST_UID}", "start_date": "2027-08-01", "end_date": "2028-05-31"},
    )
    assert response.status_code == status.HTTP_403_FORBIDDEN


def test_create_academic_hierarchy_flow():
    """Test full sequential creation of Department, Programme, Semester, Class, Subject, and Period."""
    headers = {"Authorization": f"Bearer {ADMIN_TOKEN}"}

    # 1. Create a dedicated Year for this test
    year_res = client.post(
        "/api/v1/academic/years",
        headers=headers,
        json={"name": f"HIER-{TEST_UID}", "start_date": "2026-08-01", "end_date": "2027-05-31"},
    )
    assert year_res.status_code == status.HTTP_201_CREATED
    year_id = year_res.json()["data"]["id"]

    # 2. Create Department
    dept_res = client.post(
        "/api/v1/academic/departments",
        headers=headers,
        json={"name": f"AI Dept {TEST_UID}", "code": f"D{TEST_UID[:4]}", "academic_year_id": year_id},
    )
    assert dept_res.status_code == status.HTTP_201_CREATED
    dept_id = dept_res.json()["data"]["id"]

    # 3. Create Programme
    prog_res = client.post(
        "/api/v1/academic/programmes",
        headers=headers,
        json={"name": f"B.Tech {TEST_UID}", "code": f"P{TEST_UID[:4]}", "department_id": dept_id, "duration_years": 4},
    )
    assert prog_res.status_code == status.HTTP_201_CREATED
    prog_id = prog_res.json()["data"]["id"]

    # 4. Create Semester
    sem_res = client.post(
        "/api/v1/academic/semesters",
        headers=headers,
        json={"name": "Semester 5", "number": 5, "programme_id": prog_id},
    )
    assert sem_res.status_code == status.HTTP_201_CREATED
    sem_id = sem_res.json()["data"]["id"]

    # 5. Create Class Section
    class_res = client.post(
        "/api/v1/academic/classes",
        headers=headers,
        json={"name": f"Section {TEST_UID}", "semester_id": sem_id, "max_students": 60},
    )
    assert class_res.status_code == status.HTTP_201_CREATED
    class_id = class_res.json()["data"]["id"]

    # 6. Create Subject
    sub_res = client.post(
        "/api/v1/academic/subjects",
        headers=headers,
        json={"name": f"Deep Learning {TEST_UID}", "code": f"S{TEST_UID[:4]}", "class_section_id": class_id},
    )
    assert sub_res.status_code == status.HTTP_201_CREATED
    sub_id = sub_res.json()["data"]["id"]

    # 7. Create Period
    period_res = client.post(
        "/api/v1/academic/periods",
        headers=headers,
        json={
            "name": "Period 1",
            "day_of_week": "Monday",
            "start_time": "09:00:00",
            "end_time": "09:50:00",
            "subject_id": sub_id,
            "class_section_id": class_id,
        },
    )
    assert period_res.status_code == status.HTTP_201_CREATED

    # 8. Assign Faculty to Class & Subject (BR-001)
    assign_res = client.post(
        "/api/v1/academic/assignments",
        headers=headers,
        json={"user_id": FACULTY_ID, "class_section_id": class_id, "subject_id": sub_id},
    )
    assert assign_res.status_code == status.HTTP_201_CREATED

    # 9. Duplicate Assignment should be rejected (BR-001)
    dup_assign = client.post(
        "/api/v1/academic/assignments",
        headers=headers,
        json={"user_id": FACULTY_ID, "class_section_id": class_id, "subject_id": sub_id},
    )
    assert dup_assign.status_code == status.HTTP_409_CONFLICT


def test_faculty_view_my_assigned_classes():
    """Verify Faculty can retrieve their assigned classes for attendance sessions (BR-001)."""
    response = client.get(
        "/api/v1/academic/faculty/my-classes",
        headers={"Authorization": f"Bearer {FACULTY_TOKEN}"},
    )
    assert response.status_code == status.HTTP_200_OK
    data = response.json()["data"]
    assert len(data) >= 1
