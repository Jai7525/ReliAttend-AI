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

### 3.4 Multi-Reference Biometric Core (Phase 6 & Enhancement)
- **1-to-Many Architecture:** Up to 5 natural angle references per student (`FaceReference` table with non-unique index on `student_id`).
- **Pose Metadata:** Supported angle tags: `frontal`, `slight_left`, `slight_right`, `slight_up`, `slight_down`, `custom`.
- **Detector:** OpenCV YuNet ONNX model (`face_detection_yunet_2023mar.onnx`, MIT License).
  - Strictly enforces **exactly 1 face** per reference image.
  - Computes Laplacian blur variance ($\sigma^2 \ge 50.0$) and grayscale mean brightness ($40.0 \le \mu \le 220.0$) on cropped face.
- **Recognizer:** OpenCV SFace ONNX model (`face_recognition_sface_2021dec.onnx`, Apache 2.0 License).
  - 5-point landmark face alignment (`alignCrop`).
  - Extracts 128-dimensional Float32 embeddings.
  - L2-normalization ensuring unit hypersphere representation ($\|v\|_2 = 1.0$).
  - Serialized to 512 bytes (`np.ndarray.tobytes()`).
- **Primary Reference Self-Healing:** Deleting the primary reference automatically promotes another active reference.
- **Status Reversion:** Deleting the last active reference automatically reverts the student's status to `pending` and `is_enrolled = False`.
- **Multi-Reference Recognition Pipeline:**
  - Evaluates live camera frame $v_{\text{live}}$ against reference embeddings: $s_i = v_{\text{live}} \cdot r_i$.
  - Best match: $S_{\text{final}} = \max(s_1, \dots, s_K)$.
  - *Experimental Threshold:* $0.65$ cosine similarity (labeled experimental pending field benchmarking).

---

## 4. Docker Containerisation Architecture
- **Base Image:** `python:3.13-slim-bookworm` (Official Debian minimal).
- **Runtimes:** `libglib2.0-0`, `libgomp1` (required by OpenCV headless) and `curl` (container healthcheck).
- **User Sandboxing:** Non-root execution via unprivileged `appuser` (UID 1000).
- **Volume Mounts:**
  - `./backend/models:/app/models:ro`: Read-Only bind mount supplying ONNX model weights without baking them into image layers.
  - `reliattend-data`: Named persistent volume managing SQLite database and lock files.
  - `reliattend-storage`: Named persistent volume isolating sensitive face reference images.
