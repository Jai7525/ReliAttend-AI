"""
ReliAttend AI — Health Check API Routes
Provides operational status endpoints for monitoring and testing.
"""

from datetime import datetime, timezone
from typing import Any, Dict
from fastapi import APIRouter, status
from app.config import get_settings

router = APIRouter(prefix="/health", tags=["Health"])
settings = get_settings()


@router.get(
    "",
    status_code=status.HTTP_200_OK,
    summary="Versioned API Health Check",
    description="Returns API status, operational timestamp, and environment metadata.",
    response_model=Dict[str, Any],
)
async def api_health_check() -> Dict[str, Any]:
    """Health check endpoint under API prefix."""
    return {
        "success": True,
        "message": "ReliAttend backend API v1 is running",
        "data": {
            "status": "healthy",
            "app_name": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "environment": settings.APP_ENV,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }

