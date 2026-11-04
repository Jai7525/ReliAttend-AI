"""
ReliAttend AI — Application Configuration
Loads environment variables using Pydantic Settings.
"""

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
    BIOMETRIC_MODELS_DIR: str = "./backend/models"
    BIOMETRIC_STORAGE_DIR: str = "./storage/reference_images"
    BIOMETRIC_YUNET_CONFIDENCE_THRESHOLD: float = 0.85
    BIOMETRIC_MIN_FACE_WIDTH: int = 80
    BIOMETRIC_MIN_FACE_HEIGHT: int = 80
    BIOMETRIC_BLUR_THRESHOLD: float = 50.0  # Minimum Laplacian variance
    BIOMETRIC_MIN_BRIGHTNESS: float = 40.0  # Minimum average grayscale pixel value
    BIOMETRIC_MAX_BRIGHTNESS: float = 220.0  # Maximum average grayscale pixel value

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

