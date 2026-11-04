"""
ReliAttend AI — Authentication & RBAC Automated Tests
Tests login, token generation, profile retrieval, refresh tokens, and invalid access.
"""

from fastapi import status
from fastapi.testclient import TestClient

from app.core.security import hash_password
from app.database import Base, SessionLocal, engine
from app.main import app
from app.models.user import User

client = TestClient(app)


def setup_module():
    """Ensure test users exist in database before running tests."""
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        if not db.query(User).filter_by(username="test_admin").first():
            db.add(
                User(
                    email="test_admin@reliattend.com",
                    username="test_admin",
                    password_hash=hash_password("AdminSecurePass123"),
                    full_name="Test Administrator",
                    role="admin",
                    is_active=True,
                )
            )
        if not db.query(User).filter_by(username="test_faculty").first():
            db.add(
                User(
                    email="test_faculty@reliattend.com",
                    username="test_faculty",
                    password_hash=hash_password("FacultySecurePass123"),
                    full_name="Test Faculty Member",
                    role="faculty",
                    is_active=True,
                )
            )
        db.commit()
    finally:
        db.close()


def test_login_admin_success():
    """Verify administrator login with valid credentials."""
    response = client.post(
        "/api/v1/auth/login",
        json={"username_or_email": "test_admin", "password": "AdminSecurePass123"},
    )
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["success"] is True
    assert "tokens" in data["data"]
    assert "access_token" in data["data"]["tokens"]
    assert "refresh_token" in data["data"]["tokens"]
    assert data["data"]["user"]["role"] == "admin"


def test_login_faculty_with_email_success():
    """Verify faculty login using email address."""
    response = client.post(
        "/api/v1/auth/login",
        json={"username_or_email": "test_faculty@reliattend.com", "password": "FacultySecurePass123"},
    )
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["success"] is True
    assert data["data"]["user"]["role"] == "faculty"


def test_login_wrong_password_fails():
    """Verify invalid password returns generic 401 Unauthorized."""
    response = client.post(
        "/api/v1/auth/login",
        json={"username_or_email": "test_admin", "password": "WrongPassword999"},
    )
    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    payload = response.json()
    assert payload["success"] is False


def test_login_unknown_user_fails():
    """Verify non-existent user returns 401 Unauthorized."""
    response = client.post(
        "/api/v1/auth/login",
        json={"username_or_email": "ghost_user", "password": "AnyPassword123"},
    )
    assert response.status_code == status.HTTP_401_UNAUTHORIZED


def test_get_current_user_profile():
    """Verify GET /api/v1/auth/me returns authenticated user data when valid Bearer token provided."""
    # Step 1: Login to get token
    login_res = client.post(
        "/api/v1/auth/login",
        json={"username_or_email": "test_admin", "password": "AdminSecurePass123"},
    )
    token = login_res.json()["data"]["tokens"]["access_token"]

    # Step 2: Query /me with Authorization header
    me_res = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert me_res.status_code == status.HTTP_200_OK
    user_data = me_res.json()["data"]
    assert user_data["username"] == "test_admin"
    assert user_data["role"] == "admin"


def test_get_current_user_no_token_rejected():
    """Verify GET /api/v1/auth/me rejects requests missing Authorization header."""
    response = client.get("/api/v1/auth/me")
    assert response.status_code == status.HTTP_401_UNAUTHORIZED


def test_refresh_token_endpoint():
    """Verify POST /api/v1/auth/refresh issues fresh tokens from a valid refresh token."""
    login_res = client.post(
        "/api/v1/auth/login",
        json={"username_or_email": "test_faculty", "password": "FacultySecurePass123"},
    )
    refresh_token = login_res.json()["data"]["tokens"]["refresh_token"]

    # Request new access token
    refresh_res = client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": refresh_token},
    )
    assert refresh_res.status_code == status.HTTP_200_OK
    tokens = refresh_res.json()["data"]
    assert "access_token" in tokens
    assert "refresh_token" in tokens


def test_logout_endpoint():
    """Verify POST /api/v1/auth/logout succeeds with valid Bearer token."""
    login_res = client.post(
        "/api/v1/auth/login",
        json={"username_or_email": "test_admin", "password": "AdminSecurePass123"},
    )
    token = login_res.json()["data"]["tokens"]["access_token"]

    logout_res = client.post(
        "/api/v1/auth/logout",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert logout_res.status_code == status.HTTP_200_OK
    assert logout_res.json()["success"] is True

