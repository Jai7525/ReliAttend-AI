"""
ReliAttend AI — AI Core Package Initializer
"""

from app.ai.detector import FaceDetector, FaceDetectionResult, QualityMetrics
from app.ai.recognition import FaceRecognizer, EmbeddingResult

__all__ = [
    "FaceDetector",
    "FaceDetectionResult",
    "QualityMetrics",
    "FaceRecognizer",
    "EmbeddingResult",
]
