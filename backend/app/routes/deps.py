"""
ReliAttend AI — Route Dependencies & Authorization Guards
Extracts and validates current user from JWT Bearer token and enforces RBAC.
"""

from typing import List
from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session
import jwt

from app.core.exceptions import (
    AuthenticationFailedError,
    AuthorizationForbiddenError,
)
from app.core.security import decode_token
from app.database import get_db
from app.models.user import User

# HTTP Bearer scheme for Swagger UI & API headers
security_bearer = HTTPBearer(auto_error=False)


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security_bearer),
    db: Session = Depends(get_db),
) -> User:
    """
    Extract and validate JWT Bearer token from request Authorization header.
    Returns authenticated User model or raises AuthenticationFailedError.
    """
    if not credentials:
        raise AuthenticationFailedError("Authentication credentials required. Please provide a Bearer token.")

    token = credentials.credentials
    try:
        payload = decode_token(token)
    except jwt.ExpiredSignatureError:
        raise AuthenticationFailedError("Access token has expired. Please log in or refresh your token.")
    except jwt.PyJWTError:
        raise AuthenticationFailedError("Could not validate credentials. Invalid token.")

    if payload.get("type") != "access":
        raise AuthenticationFailedError("Invalid token type. Access token expected.")

    user_id = payload.get("sub")
    if not user_id:
        raise AuthenticationFailedError("Invalid token subject.")

    user = db.query(User).filter(User.id == int(user_id)).first()
    if not user:
        raise AuthenticationFailedError("User no longer exists.")

    if not user.is_active:
        raise AuthenticationFailedError("User account has been deactivated.")

    return user


def require_role(allowed_roles: List[str]):
    """
    Dependency factory to enforce Role-Based Access Control (RBAC).
    Raises AuthorizationForbiddenError if current user role is not in allowed_roles.
    """
    def role_checker(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role not in allowed_roles:
            raise AuthorizationForbiddenError(
                f"Access denied: Operation requires one of the following roles: {', '.join(allowed_roles)}."
            )
        return current_user

    return role_checker


# Role-based dependency shortcuts
require_admin = require_role(["admin"])
require_faculty = require_role(["faculty", "admin"])  # Admin can oversee faculty operations

