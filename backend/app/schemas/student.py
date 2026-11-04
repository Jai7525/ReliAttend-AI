"""
ReliAttend AI — Student & Consent Pydantic Schemas
Defines request and response models for student profiles and biometric consent.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field


# ----------------- Consent Schemas -----------------
class ConsentRecordCreate(BaseModel):
    consent_type: str = Field(
        default="institutional_enrollment",
        description="Type of consent authorization (e.g. institutional_enrollment)",
        examples=["institutional_enrollment"],
    )
    consent_reference: Optional[str] = Field(
        None,
        description="Signed document, form ID, or approval reference",
        examples=["CONSENT-FORM-2026-001"],
    )


class ConsentRecordResponse(BaseModel):
    id: int
    student_id: int
    consent_type: str
    consent_reference: Optional[str]
    granted_at: datetime
    is_active: bool

    model_config = ConfigDict(from_attributes=True)


# ----------------- Student Schemas -----------------
class StudentCreate(BaseModel):
    student_id: str = Field(
        ...,
        description="Unique institutional student ID",
        examples=["STU2026001"],
    )
    name: str = Field(
        ...,
        min_length=2,
        description="Student full name",
        examples=["Aarav Patel"],
    )
    roll_number: str = Field(
        ...,
        description="Roll number within class section",
        examples=["26CS001"],
    )
    class_section_id: int = Field(
        ...,
        description="Class section ID where student is enrolled",
        examples=[1],
    )


class StudentUpdate(BaseModel):
    name: Optional[str] = None
    roll_number: Optional[str] = None
    class_section_id: Optional[int] = None
    is_active: Optional[bool] = None


class StudentResponse(BaseModel):
    id: int
    student_id: str
    name: str
    roll_number: str
    class_section_id: int
    registration_status: str  # pending, enrolled, deactivated
    is_active: bool
    created_at: datetime
    has_consent: bool = False
    has_face_data: bool = False

    model_config = ConfigDict(from_attributes=True)

