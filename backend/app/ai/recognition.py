"""
ReliAttend AI — Computer Vision Core (Embedding & Recognition Engine)
Independent of database and web route layers.
Uses OpenCV SFace ONNX model for face landmark alignment and feature extraction.
Produces standardized 128-dimensional L2-normalized float32 embeddings.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import os
import cv2
import numpy as np


@dataclass
class EmbeddingResult:
    embedding_bytes: bytes
    embedding_dimension: int
    model_name: str
    model_version: str
    preprocessing_version: str
    is_valid: bool
    error_message: Optional[str] = None


class FaceRecognizer:
    """
    OpenCV SFace face embedding generator.
    Enforces 128-d float32 finiteness, L2 normalization, and byte serialization.
    """

    MODEL_NAME = "opencv_sface"
    MODEL_VERSION = "2021dec"
    PREPROCESSING_VERSION = "sface_aligned_112x112_v1"
    EMBEDDING_DIM = 128

    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"SFace model weights not found at: {model_path}")

        self.model_path = model_path
        self.recognizer = cv2.FaceRecognizerSF.create(
            model=self.model_path,
            config="",
        )

    def extract_embedding(
        self, image_bgr: np.ndarray, face_data: np.ndarray
    ) -> EmbeddingResult:
        """
        Aligns and crops the face using YuNet 5-point facial landmarks,
        then extracts the 128-d float32 L2-normalized SFace embedding vector.
        """
        if image_bgr is None or face_data is None:
            return EmbeddingResult(
                embedding_bytes=b"",
                embedding_dimension=0,
                model_name=self.MODEL_NAME,
                model_version=self.MODEL_VERSION,
                preprocessing_version=self.PREPROCESSING_VERSION,
                is_valid=False,
                error_message="Invalid input image or face detection data.",
            )

        try:
            # 1. Align & crop face (standard SFace 112x112 aligned face chip)
            aligned_face = self.recognizer.alignCrop(image_bgr, face_data)
            if aligned_face is None or aligned_face.size == 0:
                return EmbeddingResult(
                    embedding_bytes=b"",
                    embedding_dimension=0,
                    model_name=self.MODEL_NAME,
                    model_version=self.MODEL_VERSION,
                    preprocessing_version=self.PREPROCESSING_VERSION,
                    is_valid=False,
                    error_message="Failed to align and crop face chip from detection landmarks.",
                )

            # 2. Extract feature embedding vector
            feature = self.recognizer.feature(aligned_face)
            if feature is None:
                return EmbeddingResult(
                    embedding_bytes=b"",
                    embedding_dimension=0,
                    model_name=self.MODEL_NAME,
                    model_version=self.MODEL_VERSION,
                    preprocessing_version=self.PREPROCESSING_VERSION,
                    is_valid=False,
                    error_message="SFace model returned empty feature vector.",
                )

            # SFace feature shape is typically (1, 128) float32
            vector = feature.flatten().astype(np.float32)

            # 3. Dimensionality validation
            if len(vector) != self.EMBEDDING_DIM:
                return EmbeddingResult(
                    embedding_bytes=b"",
                    embedding_dimension=len(vector),
                    model_name=self.MODEL_NAME,
                    model_version=self.MODEL_VERSION,
                    preprocessing_version=self.PREPROCESSING_VERSION,
                    is_valid=False,
                    error_message=f"Invalid embedding dimension {len(vector)}; expected {self.EMBEDDING_DIM}.",
                )

            # 4. Numerical validation (ensure all values are finite: no NaN or Inf)
            if not np.all(np.isfinite(vector)):
                return EmbeddingResult(
                    embedding_bytes=b"",
                    embedding_dimension=len(vector),
                    model_name=self.MODEL_NAME,
                    model_version=self.MODEL_VERSION,
                    preprocessing_version=self.PREPROCESSING_VERSION,
                    is_valid=False,
                    error_message="Embedding vector contains non-finite values (NaN or Inf).",
                )

            # 5. L2 Normalization (unit length: ||v|| = 1.0)
            norm = np.linalg.norm(vector)
            if norm == 0 or np.isnan(norm):
                return EmbeddingResult(
                    embedding_bytes=b"",
                    embedding_dimension=len(vector),
                    model_name=self.MODEL_NAME,
                    model_version=self.MODEL_VERSION,
                    preprocessing_version=self.PREPROCESSING_VERSION,
                    is_valid=False,
                    error_message="Zero-magnitude embedding vector encountered.",
                )

            normalized_vector = (vector / norm).astype(np.float32)

            # 6. Binary serialization to raw bytes
            raw_bytes = normalized_vector.tobytes()

            return EmbeddingResult(
                embedding_bytes=raw_bytes,
                embedding_dimension=self.EMBEDDING_DIM,
                model_name=self.MODEL_NAME,
                model_version=self.MODEL_VERSION,
                preprocessing_version=self.PREPROCESSING_VERSION,
                is_valid=True,
                error_message=None,
            )

        except Exception as exc:
            return EmbeddingResult(
                embedding_bytes=b"",
                embedding_dimension=0,
                model_name=self.MODEL_NAME,
                model_version=self.MODEL_VERSION,
                preprocessing_version=self.PREPROCESSING_VERSION,
                is_valid=False,
                error_message=f"Feature extraction failed: {str(exc)}",
            )

    @staticmethod
    def deserialize_embedding(raw_bytes: bytes) -> np.ndarray:
        """
        Safely unpacks binary BLOB from database back into a 128-d float32 numpy array.
        """
        if not raw_bytes or len(raw_bytes) != 128 * 4:  # 128 float32 = 512 bytes
            raise ValueError(f"Invalid embedding byte length {len(raw_bytes) if raw_bytes else 0}; expected 512 bytes.")
        return np.frombuffer(raw_bytes, dtype=np.float32)

