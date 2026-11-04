"""
ReliAttend AI — Multi-Reference Biometric Schemas (Phase 6 Enhancement)
Defines request and response schemas supporting up to 5 reference images per student.
Strictly safeguards privacy: zero filesystem paths and zero raw embeddings exposed.
"""

from datetime import datetime
from typing import List, Optional
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


class FaceReferenceItemResponse(BaseModel):
    reference_id: int
    pose_label: str
    is_primary: bool
    status: str
    quality_score: Optional[float] = None
    model_name: str
    embedding_dimension: int
    enrolled_at: datetime

    model_config = ConfigDict(from_attributes=True)


class FaceRegistrationResponse(BaseModel):
    """
    Response returned when an individual reference image is added or registered.
    Maintains full backward-compatibility with existing registration API callers.
    """
    success: bool = True
    student_id: str
    student_name: str
    registration_id: int
    registration_status: str
    face_count: int
    quality_status: str
    quality_metrics: QualityMetricsResponse
    pose_label: str = "frontal"
    is_primary: bool = True
    active_reference_count: int = 1
    model_name: str
    embedding_dimension: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class FaceRegistrationStatusResponse(BaseModel):
    """
    Comprehensive biometric status response.
    Reports total references, active reference count (up to 5), and enrolled angles.
    """
    student_id: str
    student_name: str
    is_enrolled: bool
    reference_count: int
    active_reference_count: int
    registration_status: str
    primary_reference_id: Optional[int] = None
    active_poses: List[str] = Field(default_factory=list, description="List of angles/poses enrolled")
    model_name: Optional[str] = None
    embedding_dimension: Optional[int] = None
    latest_registration_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class StudentReferenceListResponse(BaseModel):
    """
    Lists all active reference items for a student.
    Never exposes internal filesystem paths or raw embedding vectors.
    """
    student_id: str
    student_name: str
    active_count: int
    max_allowed: int
    references: List[FaceReferenceItemResponse]

    model_config = ConfigDict(from_attributes=True)
