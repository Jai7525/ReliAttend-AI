"""
ReliAttend AI — API Routers Aggregator
"""

from fastapi import APIRouter
from app.routes.health import router as health_router
from app.routes.auth import router as auth_router
from app.routes.academic import router as academic_router
from app.routes.students import router as students_router
from app.routes.biometrics import router as biometrics_router
from app.routes.detection import router as detection_router
from app.routes.recognition import router as recognition_router
from app.routes.tracking import router as tracking_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(auth_router)
api_router.include_router(academic_router)
api_router.include_router(students_router)
api_router.include_router(biometrics_router)
api_router.include_router(detection_router)
api_router.include_router(recognition_router)
api_router.include_router(tracking_router)
