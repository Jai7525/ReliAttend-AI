"""
ReliAttend AI — Multi-Face Recognition Schemas (Phase 8)
Defines request and response models for 1:N face matching in classroom frames.
Privacy by design: Never exposes filesystem paths or raw embedding vectors.
"""

from typing import List, Optional
from pydantic import BaseModel, Field

from app.schemas.detection import BoundingBoxSchema, FaceQualitySchema, FrameMetadataSchema


class MatchCandidateSchema(BaseModel):
    student_id: str = Field(..., description="Institutional student ID e.g. 'STU2026001'")
    student_name: str = Field(..., description="Student full name")
    similarity_score: float = Field(..., description="Cosine similarity score (0.0 to 1.0)")
    matched_pose: str = Field(..., description="Enrolled pose variation that matched best")
    is_primary: bool = Field(..., description="Whether the matched reference was primary")


class RecognizedFaceItemSchema(BaseModel):
    face_id: int = Field(..., description="1-indexed sequence ID within the frame")
    bounding_box: BoundingBoxSchema = Field(..., description="Face coordinates in original frame pixels")
    detection_confidence: float = Field(..., description="YuNet detection confidence")
    quality: FaceQualitySchema = Field(..., description="Per-face image quality indicators")
    match_status: str = Field(..., description="'matched', 'uncertain', or 'unrecognized'")
    top_match: Optional[MatchCandidateSchema] = Field(None, description="Best candidate match if applicable")
    top_candidates: List[MatchCandidateSchema] = Field(default_factory=list, description="Top-K candidate matches")
    margin_to_second: Optional[float] = Field(None, description="Score margin between top-1 and top-2 distinct candidates, or null if fewer than 2 candidates exist")
    reason: Optional[str] = Field(None, description="Diagnostic explanation for uncertain or unrecognized results")


class RecognitionSummarySchema(BaseModel):
    section_id: int = Field(..., description="Class section ID")
    section_name: str = Field(..., description="Class section name")
    faces_detected: int = Field(..., description="Total faces detected in frame")
    faces_recognized: int = Field(..., description="Total faces with status 'matched'")
    faces_uncertain: int = Field(..., description="Total faces with status 'uncertain'")
    faces_unrecognized: int = Field(..., description="Total faces with status 'unrecognized'")
    gallery_students_count: int = Field(..., description="Total enrolled students with active consent in gallery")
    gallery_references_count: int = Field(..., description="Total active reference embeddings in gallery")


class FrameRecognitionData(BaseModel):
    summary: RecognitionSummarySchema
    frame_metadata: FrameMetadataSchema
    faces: List[RecognizedFaceItemSchema] = Field(default_factory=list)


class FrameRecognitionResponse(BaseModel):
    success: bool = True
    message: str = "Multi-face recognition completed successfully."
    data: FrameRecognitionData

