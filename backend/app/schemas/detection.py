"""
ReliAttend AI — Multi-Face Detection Schemas (Phase 7)
Defines request and response schemas for multi-face detection in classroom frames.
Privacy by design: returns coordinates and quality indicators; does not identify students.
"""

from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, ConfigDict, Field


class BoundingBoxSchema(BaseModel):
    x: int = Field(..., description="Top-left X coordinate in original frame pixels")
    y: int = Field(..., description="Top-left Y coordinate in original frame pixels")
    width: int = Field(..., description="Bounding box width in original frame pixels")
    height: int = Field(..., description="Bounding box height in original frame pixels")


class FacialLandmarksSchema(BaseModel):
    right_eye: Tuple[int, int] = Field(..., description="Right eye coordinate [x, y]")
    left_eye: Tuple[int, int] = Field(..., description="Left eye coordinate [x, y]")
    nose_tip: Tuple[int, int] = Field(..., description="Nose tip coordinate [x, y]")
    mouth_right: Tuple[int, int] = Field(..., description="Right mouth corner coordinate [x, y]")
    mouth_left: Tuple[int, int] = Field(..., description="Left mouth corner coordinate [x, y]")


class FaceQualitySchema(BaseModel):
    blur_score: float = Field(..., description="Laplacian variance on face crop")
    brightness_score: float = Field(..., description="Mean grayscale intensity on face crop")
    face_width: int = Field(..., description="Face width in pixels")
    face_height: int = Field(..., description="Face height in pixels")
    is_blur_acceptable: bool = Field(..., description="Whether blur threshold was satisfied")
    is_brightness_acceptable: bool = Field(..., description="Whether brightness bounds were satisfied")
    is_resolution_acceptable: bool = Field(..., description="Whether minimum resolution was satisfied")
    is_acceptable: bool = Field(..., description="Overall quality acceptability status")


class DetectedFaceSchema(BaseModel):
    face_id: int = Field(..., description="1-indexed sequence ID within the frame")
    bounding_box: BoundingBoxSchema = Field(..., description="Coordinates in original frame pixels")
    confidence: float = Field(..., description="YuNet detector confidence score (0.0 to 1.0)")
    landmarks: FacialLandmarksSchema = Field(..., description="5 facial landmarks in original frame pixels")
    quality: FaceQualitySchema = Field(..., description="Per-face image quality indicators")


class FrameMetadataSchema(BaseModel):
    width: int = Field(..., description="Original image width in pixels")
    height: int = Field(..., description="Original image height in pixels")
    channels: int = Field(default=3, description="Color channels count")
    inference_time_ms: float = Field(..., description="Detection inference elapsed time in milliseconds")
    is_scaled: bool = Field(default=False, description="Whether frame was downscaled for inference")
    scale_factor: float = Field(default=1.0, description="Scale factor applied during inference")


class MultiFaceDetectionData(BaseModel):
    face_count: int = Field(..., description="Total number of detected faces")
    frame_metadata: FrameMetadataSchema = Field(..., description="Input image spatial metadata")
    faces: List[DetectedFaceSchema] = Field(default_factory=list, description="List of detected faces")


class MultiFaceDetectionResponse(BaseModel):
    success: bool = True
    message: str = "Detection completed successfully."
    data: MultiFaceDetectionData

