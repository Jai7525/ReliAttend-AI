"""
ReliAttend AI — Custom Application Exceptions
Provides uniform error definitions and handlers across the application.
"""

from typing import Any, Dict, Optional
from fastapi import HTTPException, status


class ReliAttendBaseException(HTTPException):
    """Base exception for ReliAttend application."""

    def __init__(
        self,
        status_code: int,
        message: str,
        errors: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            status_code=status_code,
            detail={
                "success": False,
                "message": message,
                "errors": errors or {},
            },
        )


class ResourceNotFoundError(ReliAttendBaseException):
    """Raised when a requested resource does not exist."""

    def __init__(self, message: str = "Resource not found"):
        super().__init__(status_code=status.HTTP_404_NOT_FOUND, message=message)


class AuthenticationFailedError(ReliAttendBaseException):
    """Raised when authentication credentials cannot be validated."""

    def __init__(self, message: str = "Invalid credentials"):
        super().__init__(status_code=status.HTTP_401_UNAUTHORIZED, message=message)


class AuthorizationForbiddenError(ReliAttendBaseException):
    """Raised when the user does not have permission for the operation."""

    def __init__(self, message: str = "Operation not permitted"):
        super().__init__(status_code=status.HTTP_403_FORBIDDEN, message=message)


class DuplicateResourceError(ReliAttendBaseException):
    """Raised when an entity with unique constraints already exists."""

    def __init__(self, message: str = "Resource already exists"):
        super().__init__(status_code=status.HTTP_409_CONFLICT, message=message)


class ValidationProcessingError(ReliAttendBaseException):
    """Raised when request payload or data validation fails business rules."""

    def __init__(self, message: str = "Validation failed", errors: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            message=message,
            errors=errors,
        )


class BadRequestError(ReliAttendBaseException):
    """Raised when the client sends a malformed or invalid request."""

    def __init__(self, message: str = "Bad request"):
        super().__init__(status_code=status.HTTP_400_BAD_REQUEST, message=message)

