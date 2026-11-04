# ReliAttend AI — AI Model Licensing & Compatibility Review

This document records the exact pretrained model files, download sources, versions, code and model weight licenses, and compatibility evaluations for the Phase 6 Biometric Core.

---

## 1. Candidate Evaluation Summary

| Candidate Pipeline | Face Detection | Feature Extraction / Recognition | Python 3.13 Windows Wheel Availability | Licensing & Weight Constraints | Decision |
|---|---|---|---|---|---|
| **YuNet + SFace** | OpenCV YuNet (`face_detection_yunet`) | OpenCV SFace (`face_recognition_sface`) | **Yes (via official `opencv-python-headless`)** | MIT (YuNet) / Apache 2.0 (SFace) — Permissive for academic & commercial | **SELECTED** |
| **InsightFace (ArcFace)** | RetinaFace | ArcFace (Buffalo_L) | No official Python 3.13 prebuilt wheel; high compile risk on Windows | Weights strictly non-commercial research use only | Rejected |
| **Dlib + face_recognition** | HOG / CNN | ResNet-34 (Davis King) | No official Python 3.13 wheel on Windows (requires C++ CMake build) | Public Domain / CC0 weights | Rejected (Build failure risk) |
| **FaceNet** | MTCNN | Inception-ResNet | Requires heavy PyTorch / TensorFlow stack (~1.5GB) | Varied; datasets frequently non-commercial (VGGFace2/CASIA) | Rejected (Footprint too large) |

---

## 2. Selected Models — Detailed Verification

### A. Face Detection: YuNet
- **Model Name:** YuNet (`face_detection_yunet`)
- **Version:** `2023mar` (Optimized for OpenCV 4.x/5.x DNN engine)
- **Official Model File:** `face_detection_yunet_2023mar.onnx`
- **Model Weight Size:** ~335 KB
- **Author / Institution:** Shiqi Yu (Southern University of Science and Technology, `libfacedetection`)
- **Official Repository:** [OpenCV Zoo YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
- **Direct Download URL:**
  `https://github.com/opencv/opencv_zoo/raw/refs/heads/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx`
- **Code License:** MIT License
- **Pretrained Weights License:** MIT License (Permissive, commercial and academic usage permitted)

### B. Face Recognition & Embedding: SFace
- **Model Name:** SFace (`face_recognition_sface`)
- **Version:** `2021dec` (Zhong et al., "SFace: Sigmoid-Constrained Hypersphere Loss for Robust Face Recognition", IEEE TIP 2021)
- **Official Model File:** `face_recognition_sface_2021dec.onnx`
- **Model Weight Size:** ~37.4 MB
- **Embedding Dimension:** 128-dimensional Float32 vector
- **Normalization:** L2-normalized unit sphere ($\|v\|_2 = 1.0$)
- **Official Repository:** [OpenCV Zoo SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface)
- **Direct Download URL:**
  `https://github.com/opencv/opencv_zoo/raw/refs/heads/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx`
- **Code License:** Apache License 2.0
- **Pretrained Weights License:** Apache License 2.0 (Permissive, commercial and academic usage permitted)

---

## 3. Storage and Tracking Policy
- Pretrained `.onnx` model files must **never be committed to Git**.
- Model weights directory: `backend/models/`
- Added to `.gitignore`: `backend/models/*.onnx`
- Checksums (SHA-256) will be verified during automated initialization or test setup.

