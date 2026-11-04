"""
ReliAttend AI — Authentication & User Pydantic Schemas
Defines request and response contracts for auth operations.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict, EmailStr, Field


class LoginRequest(BaseModel):
    """Credentials payload for user authentication."""
    username_or_email: str = Field(
        ...,
        description="User email address or unique username",
        examples=["admin@reliattend.com", "admin"],
    )
    password: str = Field(
        ...,
        min_length=6,
        description="Plaintext password",
        examples=["AdminPassword@123"],
    )


class RefreshTokenRequest(BaseModel):
    """Payload to refresh an expired access token."""
    refresh_token: str = Field(
        ...,
        description="Valid JWT refresh token",
    )


class TokenResponse(BaseModel):
    """Authentication tokens returned upon successful login or refresh."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = Field(description="Access token lifespan in seconds")


class UserResponse(BaseModel):
    """Public user profile data returned to client."""
    id: int
    email: str
    username: str
    full_name: str
    role: str
    is_active: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

