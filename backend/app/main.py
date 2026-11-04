"""
ReliAttend AI — Backend Application Entry Point
FastAPI application initialization, middleware configuration, and router binding.
"""

from contextlib import asynccontextmanager
import logging
from typing import Any, Dict
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.core.exceptions import ReliAttendBaseException
from app.routes import api_router

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("reliattend.app")

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown routines."""
    logger.info("Starting %s v%s in %s mode", settings.APP_NAME, settings.APP_VERSION, settings.APP_ENV)
    yield
    logger.info("Shutting down %s", settings.APP_NAME)


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Confidence-Aware Multi-Face Recognition and Reliability-Based Smart Attendance Management System Backend.",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global Exception Handler for custom application exceptions
@app.exception_handler(ReliAttendBaseException)
async def reliattend_exception_handler(request: Request, exc: ReliAttendBaseException):
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.detail,
    )


# Root Health Check Endpoint
@app.get(
    "/health",
    tags=["Root"],
    summary="Root Service Health Check",
    description="Quick ping endpoint to verify that the server is alive.",
    response_model=Dict[str, Any],
)
async def root_health():
    return {
        "status": "healthy",
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.APP_ENV,
    }


# Include versioned API router (/api/v1/...)
app.include_router(api_router, prefix=settings.API_V1_PREFIX)

