"""
ReliAttend AI — Authentication Endpoints Router
Implements user login, current profile verification, token refresh, and logout.
"""

from typing import Any, Dict
from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.user import User
from app.routes.deps import get_current_user
from app.schemas.auth import (
    LoginRequest,
    RefreshTokenRequest,
    TokenResponse,
    UserResponse,
)
from app.services.auth_service import AuthService

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post(
    "/login",
    status_code=status.HTTP_200_OK,
    summary="User Login",
    description="Authenticate user with email or username and password, returning JWT access and refresh tokens.",
)
async def login(
    payload: LoginRequest,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Authenticate credentials and generate tokens."""
    user = AuthService.authenticate_user(
        db=db,
        identifier=payload.username_or_email,
        password=payload.password,
    )
    tokens = AuthService.create_user_tokens(user=user)
    return {
        "success": True,
        "message": "Authentication successful",
        "data": {
            "tokens": tokens.model_dump(),
            "user": UserResponse.model_validate(user).model_dump(),
        },
    }


@router.post(
    "/refresh",
    status_code=status.HTTP_200_OK,
    summary="Refresh Access Token",
    description="Exchange a valid refresh token for a fresh access token without re-authenticating.",
)
async def refresh_token(
    payload: RefreshTokenRequest,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Issue a fresh access token using a refresh token."""
    tokens = AuthService.refresh_access_token(
        db=db,
        refresh_token_str=payload.refresh_token,
    )
    return {
        "success": True,
        "message": "Access token refreshed successfully",
        "data": tokens.model_dump(),
    }


@router.get(
    "/me",
    status_code=status.HTTP_200_OK,
    summary="Get Current User Profile",
    description="Retrieve profile details and role of the currently authenticated user.",
)
async def get_current_user_profile(
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """Return authenticated user profile data."""
    return {
        "success": True,
        "message": "User profile retrieved successfully",
        "data": UserResponse.model_validate(current_user).model_dump(),
    }


@router.post(
    "/logout",
    status_code=status.HTTP_200_OK,
    summary="User Logout",
    description="Log out user and signal client-side session termination.",
)
async def logout(
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """Terminate user session."""
    return {
        "success": True,
        "message": "Logged out successfully",
        "data": {
            "user_id": current_user.id,
            "username": current_user.username,
        },
    }

