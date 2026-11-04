"""
ReliAttend AI — Biometric & Face Registration Schemas
Defines request and response schemas for Phase 6.
Guarantees NO raw embeddings or server filesystem paths are exposed in responses.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field


class QualityMetricsResponse(BaseModel):
    blur_score: float = Field(..., description="Laplacian variance on face crop")
    brightness_score: float = Field(..., description="Mean grayscale intensity on face crop")
    face_width: int = Field(..., description="Face width in pixels")
    face_height: int = Field(..., description="Face height in pixels")
    detection_confidence: float = Field(..., description="YuNet detector confidence score")
    is_blur_acceptable: bool = Field(..., description="Whether blur threshold was satisfied")
    is_brightness_acceptable: bool = Field(..., description="Whether brightness bounds were satisfied")
    is_resolution_acceptable: bool = Field(..., description="Whether minimum resolution was satisfied")


class FaceRegistrationResponse(BaseModel):
    success: bool = True
    student_id: str
    student_name: str
    registration_id: int
    registration_status: str
    face_count: int
    quality_status: str
    quality_metrics: QualityMetricsResponse
    model_name: str
    embedding_dimension: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class FaceRegistrationStatusResponse(BaseModel):
    student_id: str
    student_name: str
    is_enrolled: bool
    reference_count: int
    active_reference_count: int
    registration_status: str
    model_name: Optional[str] = None
    embedding_dimension: Optional[int] = None
    latest_registration_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)

