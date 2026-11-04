# ReliAttend AI — Architecture Documentation

## 1. System Overview
ReliAttend AI is a Confidence-Aware Multi-Face Recognition and Reliability-Based Smart Attendance Management System designed to eliminate proxy attendance and manual errors in educational environments.

---

## 2. Core Architectural Principles
1. **Modular Monolithic Backend:** Clean separation between models, schemas, business services, and AI vision modules.
2. **Strict Scoped RBAC:** Enforces role-based boundaries. Faculty can only view and manage students and attendance in their explicitly assigned classes (`FacultyAssignment`).
3. **Biometric Privacy by Design:**
   - Raw face embeddings (128-d float32) are stored as binary BLOBs inside SQLite/PostgreSQL.
   - Reference images are assigned random UUID filenames and saved in protected local storage (`storage/reference_images/`).
   - Server internal paths and binary embeddings are **never exposed in API outputs**.
   - Biometric enrolment requires prior active consent (`ConsentRecord`, BR-018).
4. **Resilient AI Pipeline:**
   - Pretrained model weights are decoupled from code and tracked via clear Apache 2.0 / MIT licensing.
   - Quality checks (blur, brightness, resolution) prevent degraded representations.

---

## 3. Subsystem Breakdown

### 3.1 Authentication & RBAC (Phase 3)
- Signed JSON Web Tokens (HS256) with access tokens (30 min) and refresh tokens (7 days).
- Secure password hashing via direct `bcrypt`.

### 3.2 Academic Hierarchy (Phase 4)
- Enforces relational tree: `AcademicYear` $\rightarrow$ `Department` $\rightarrow$ `Programme` $\rightarrow$ `Semester` $\rightarrow$ `ClassSection` $\rightarrow$ `Subject` $\rightarrow$ `Period`.
- Assigns faculty via `FacultyAssignment` with unique constraint per `(faculty_id, subject_id, section_id)`.

### 3.3 Student Management (Phase 5)
- Manages institutional IDs, names, and section assignments.
- Enforces unique institutional ID institution-wide and unique roll number within a section.
- Tracks active biometric consent.

### 3.4 Biometric Enrolment & Feature Extraction Core (Phase 6)
- **Detector:** OpenCV YuNet ONNX model (`face_detection_yunet_2023mar.onnx`, MIT License).
  - Strictly enforces **exactly 1 face** per reference image.
  - Computes Laplacian blur variance ($\sigma^2$) and grayscale mean brightness on the cropped face region.
- **Recognizer:** OpenCV SFace ONNX model (`face_recognition_sface_2021dec.onnx`, Apache 2.0 License).
  - 5-point landmark face alignment (`alignCrop`).
  - Extracts 128-dimensional Float32 embeddings.
  - L2-normalization ensuring unit hypersphere representation ($\|v\|_2 = 1.0$).
  - Serialized to 512 bytes (`np.ndarray.tobytes()`).
- **Replacement Policy:** One active reference per student. Re-registration updates the active record and safely removes superseded images from disk.
- **Atomic Rollback:** Cleans up file artifacts if database transactions fail.

---

## 4. Database ER Structure (Summary of Key Tables)
- `users`: Core authentication identity.
- `students`: Institutional student profiles.
- `consent_records`: Biometric authorization audit records.
- `face_references`: Encrypted/binary BLOB embeddings and reference image pointers.
- `audit_logs`: Immutable security and operation audit trail.

---

## 5. Docker Containerisation Architecture
- **Base Image:** `python:3.13-slim-bookworm` (Official Debian minimal).
- **Runtimes:** `libglib2.0-0`, `libgomp1` (required by OpenCV headless) and `curl` (container healthcheck).
- **User Sandboxing:** Non-root execution via unprivileged `appuser` (UID 1000).
- **Volume Mounts:**
  - `./backend/models:/app/models:ro`: Read-Only bind mount supplying ONNX model weights without baking them into image layers.
  - `reliattend-data`: Named persistent volume managing SQLite database and lock files.
  - `reliattend-storage`: Named persistent volume isolating sensitive face reference images.

