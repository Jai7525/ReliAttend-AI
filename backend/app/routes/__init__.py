"""
ReliAttend AI — API Routers Aggregator
"""

from fastapi import APIRouter
from app.routes.health import router as health_router
from app.routes.auth import router as auth_router
from app.routes.academic import router as academic_router
from app.routes.students import router as students_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(auth_router)
api_router.include_router(academic_router)
api_router.include_router(students_router)
