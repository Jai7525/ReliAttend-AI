"""
ReliAttend AI — Application Configuration
Loads environment variables using Pydantic Settings.
"""

import os
from functools import lru_cache
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # App Settings
    APP_NAME: str = "ReliAttend AI Backend"
    APP_VERSION: str = "1.0.0"
    APP_ENV: str = "development"
    DEBUG: bool = True
    API_V1_PREFIX: str = "/api/v1"
    HOST: str = "127.0.0.1"
    PORT: int = 8000

    # CORS Settings
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]

    # Security (For Phase 3)
    SECRET_KEY: str = "development-secret-key-change-in-production-reliattend-2026"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # Database (For Phase 2)
    DATABASE_URL: str = "sqlite:///./reliattend.db"

    # Storage Paths
    STORAGE_PATH: str = "./storage"

    # Biometric Image & Face Processing Settings (Phase 6)
    # Note: Quality thresholds are initial experimental defaults, fully configurable
    BIOMETRIC_MAX_UPLOAD_BYTES: int = 5 * 1024 * 1024  # 5 MB limit
    BIOMETRIC_ALLOWED_MIME_TYPES: List[str] = ["image/jpeg", "image/png"]
    BIOMETRIC_MODELS_DIR: str = os.getenv(
        "BIOMETRIC_MODELS_DIR",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    )
    BIOMETRIC_STORAGE_DIR: str = os.getenv(
        "BIOMETRIC_STORAGE_DIR",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "storage", "reference_images")
    )
    BIOMETRIC_YUNET_CONFIDENCE_THRESHOLD: float = 0.85
    BIOMETRIC_MIN_FACE_WIDTH: int = 80
    BIOMETRIC_MIN_FACE_HEIGHT: int = 80
    BIOMETRIC_BLUR_THRESHOLD: float = 50.0  # Minimum Laplacian variance
    BIOMETRIC_MIN_BRIGHTNESS: float = 40.0  # Minimum average grayscale pixel value
    BIOMETRIC_MAX_BRIGHTNESS: float = 220.0  # Maximum average grayscale pixel value

    # Multi-Reference Enrollment Settings (Phase 6 Enhancement)
    BIOMETRIC_MAX_ACTIVE_REFERENCES: int = 5
    BIOMETRIC_MIN_ACTIVE_REFERENCES: int = 1
    BIOMETRIC_ALLOWED_POSES: List[str] = [
        "frontal",
        "slight_left",
        "slight_right",
        "slight_up",
        "slight_down",
        "custom",
    ]
    # Experimental similarity threshold for SFace multi-reference cosine matching
    # (Treated as experimental; not a finalized or hardcoded production threshold)
    BIOMETRIC_EXPERIMENTAL_MATCH_THRESHOLD: float = 0.65

    # Multi-Face Group Frame Detection Settings (Phase 7)
    DETECTION_DEFAULT_CONFIDENCE_THRESHOLD: float = 0.60
    DETECTION_DEFAULT_NMS_THRESHOLD: float = 0.30
    DETECTION_MAX_FACES_PER_FRAME: int = 150  # Upper bound safety cap for dense classroom groups
    DETECTION_MAX_FRAME_WIDTH: int = 4096     # 4K max dimension safety limit
    DETECTION_MAX_FRAME_HEIGHT: int = 4096    # 4K max dimension safety limit
    DETECTION_INFERENCE_MAX_DIMENSION: int = 1920  # Max dimension for detector scaling (retains aspect ratio)

    # Multi-Face Recognition & 1:N Matching Settings (Phase 8)
    # Note: Thresholds are initial experimental defaults and subject to empirical calibration
    RECOGNITION_EXPERIMENTAL_MATCH_THRESHOLD: float = 0.65
    RECOGNITION_EXPERIMENTAL_AMBIGUITY_MARGIN: float = 0.05
    RECOGNITION_TOP_K: int = 3
    RECOGNITION_ALLOW_DUPLICATE_ASSIGNMENTS: bool = False  # Flags multiple faces matching same student as uncertain

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


@lru_cache()
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()

