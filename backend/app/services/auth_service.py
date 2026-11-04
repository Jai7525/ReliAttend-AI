"""
ReliAttend AI — Authentication Business Logic Service
Handles user verification, credential validation, and JWT token lifecycle.
"""

from datetime import timedelta
import logging
from sqlalchemy import or_
from sqlalchemy.orm import Session
import jwt

from app.config import get_settings
from app.core.exceptions import AuthenticationFailedError
from app.core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    verify_password,
)
from app.models.user import User
from app.schemas.auth import TokenResponse

logger = logging.getLogger("reliattend.auth")
settings = get_settings()


class AuthService:
    """Authentication and identity verification service."""

    @staticmethod
    def authenticate_user(db: Session, identifier: str, password: str) -> User:
        """
        Authenticate a user by username or email.
        Raises AuthenticationFailedError if credentials do not match.
        """
        clean_identifier = identifier.strip().lower()
        user = db.query(User).filter(
            or_(
                User.email == clean_identifier,
                User.username == clean_identifier,
            )
        ).first()

        if not user:
            logger.warning("Authentication failed: User '%s' not found", identifier)
            raise AuthenticationFailedError("Invalid email/username or password")

        if not user.is_active:
            logger.warning("Authentication failed: User '%s' is deactivated", identifier)
            raise AuthenticationFailedError("User account has been deactivated")

        if not verify_password(password, user.password_hash):
            logger.warning("Authentication failed: Incorrect password for user '%s'", identifier)
            raise AuthenticationFailedError("Invalid email/username or password")

        logger.info("User '%s' (ID: %s, Role: %s) authenticated successfully", user.username, user.id, user.role)
        return user

    @staticmethod
    def create_user_tokens(user: User) -> TokenResponse:
        """Generate access and refresh tokens for authenticated user."""
        access_token = create_access_token(
            user_id=user.id,
            username=user.username,
            role=user.role,
        )
        refresh_token = create_refresh_token(user_id=user.id)
        expires_in = settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60

        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=expires_in,
        )

    @staticmethod
    def refresh_access_token(db: Session, refresh_token_str: str) -> TokenResponse:
        """Validate refresh token and issue a fresh access token."""
        try:
            payload = decode_token(refresh_token_str)
        except jwt.ExpiredSignatureError:
            raise AuthenticationFailedError("Refresh token has expired. Please log in again.")
        except jwt.PyJWTError:
            raise AuthenticationFailedError("Invalid refresh token.")

        if payload.get("type") != "refresh":
            raise AuthenticationFailedError("Token provided is not a refresh token.")

        user_id = payload.get("sub")
        if not user_id:
            raise AuthenticationFailedError("Invalid token subject.")

        user = db.query(User).filter(User.id == int(user_id)).first()
        if not user or not user.is_active:
            raise AuthenticationFailedError("User account no longer active.")

        # Issue fresh access token and retain or rotate refresh token
        new_access_token = create_access_token(
            user_id=user.id,
            username=user.username,
            role=user.role,
        )
        new_refresh_token = create_refresh_token(user_id=user.id)
        expires_in = settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60

        return TokenResponse(
            access_token=new_access_token,
            refresh_token=new_refresh_token,
            token_type="bearer",
            expires_in=expires_in,
        )

