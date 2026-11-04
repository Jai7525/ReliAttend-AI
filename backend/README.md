# ReliAttend AI — Backend Service

Modular FastAPI backend service for the **ReliAttend AI: Confidence-Aware Multi-Face Recognition and Reliability-Based Smart Attendance Management System**.

---

## 1. Directory Structure

```
backend/
├── app/
│   ├── ai/               # Computer vision & face recognition core (YuNet & SFace)
│   │   ├── detector.py   # OpenCV YuNet detection & face-region quality metrics
│   │   ├── recognition.py# OpenCV SFace 128-d L2-normalized feature extraction
│   │   └── __init__.py
│   ├── core/             # Security, JWT tokens, exception definitions
│   │   ├── exceptions.py
│   │   ├── security.py
│   │   └── __init__.py
│   ├── models/           # SQLAlchemy 2.0 ORM models
│   │   ├── academic.py   # Years, Departments, Programmes, Semesters, Sections, Subjects, FacultyAssignments
│   │   ├── attendance.py # Sessions, Evidence, Records, AuditLogs
│   │   ├── student.py    # Students, ConsentRecords, FaceReferences (binary BLOB embeddings)
│   │   ├── user.py       # Users (Admin & Faculty RBAC)
│   │   └── __init__.py
│   ├── routes/           # FastAPI REST API route endpoints
│   │   ├── academic.py
│   │   ├── auth.py
│   │   ├── biometrics.py # Reference face registration & status inquiries (Phase 6)
│   │   ├── deps.py       # Authentication & scoped RBAC guards
│   │   ├── health.py
│   │   ├── students.py
│   │   └── __init__.py
│   ├── schemas/          # Pydantic v2 data validation schemas
│   │   ├── academic.py
│   │   ├── auth.py
│   │   ├── biometrics.py
│   │   ├── student.py
│   │   └── __init__.py
│   ├── services/         # Business & domain services
│   │   ├── academic_service.py
│   │   ├── auth_service.py
│   │   ├── biometric_service.py # Enrolment orchestration, consent validation, rollback safety
│   │   ├── student_service.py
│   │   └── __init__.py
│   ├── config.py         # Application settings (Pydantic BaseSettings)
│   ├── database.py       # SQLAlchemy engine & session factory
│   └── main.py           # FastAPI entrypoint, middleware, lifespan
├── models/               # Local ONNX weights (gitignored: YuNet & SFace)
├── postman/              # Postman collection (ReliAttend_API.json)
├── tests/                # Automated pytest test suites
└── requirements.txt
```

---

## 2. Phase 6 Biometric Core: Architecture & Policies

### 2.1 AI Models & Licensing
- **Face Detection:** OpenCV YuNet (`face_detection_yunet_2023mar.onnx`, ~232 KB).
  - *Weights & Code License:* **MIT License** (Shiqi Yu / libfacedetection).
- **Face Feature Extraction:** OpenCV SFace (`face_recognition_sface_2021dec.onnx`, ~38.6 MB).
  - *Weights & Code License:* **Apache License 2.0** (Zhong et al., IEEE TIP 2021).
  - *Embedding Vector:* Exactly **128-dimensional Float32 vector**, unit L2-normalized ($\|v\|_2 = 1.0$).
- Full details documented in [`docs/ai-model-licences.md`](file:///c:/Users/JAI/Downloads/ReliAttend-AI/docs/ai-model-licences.md).

### 2.2 Biometric Consent Enforcement (BR-018 & FR-ADMIN-004)
- Biometric registration is **strictly blocked** if no active `ConsentRecord` exists for the student.
- Rejection returns `403 Forbidden` (`Biometric registration blocked: Student does not have an active consent record on file`).

### 2.3 Single-Face & Face-Region Quality Validation
Every uploaded reference image must satisfy:
1. **Magic-Byte Verification:** Verifies JPEG (`FF D8 FF`) or PNG (`89 50 4E 47`) file signatures before decoding.
2. **Exactly-One-Face Rule:** Rejects images with 0 faces (`422 Unprocessable`) or $>1$ face (`422 Unprocessable`).
3. **Face-Region Blur:** Computes Laplacian variance on cropped face ($\sigma^2 \ge \text{threshold}$, default 50.0).
4. **Face-Region Brightness:** Computes mean intensity on cropped face ($40.0 \le \text{brightness} \le 220.0$).
5. **Face Resolution:** Face bounding box must be at least $80 \times 80$ pixels.
- *All thresholds are configurable via environment variables in `config.py`.*

### 2.4 Scoped RBAC Permissions
- **System Administrators:** Full institutional access across all students and departments.
- **Faculty:** Scoped strictly via `FacultyAssignment`. A faculty member can only register reference faces or query status for students enrolled in class sections assigned to them. Cross-class access is rejected with `403 Forbidden`.

### 2.5 Storage, Replacement & Rollback Policies
- **Image Storage:** Reference face photos are stored under `storage/reference_images/{uuid4}.jpg` outside the web root. Files are strictly gitignored.
- **Data Protection:** Server file paths and raw embeddings are **never returned in API responses** or exposed in logs.
- **Single Active Reference:** Re-registration updates the active reference in-place, safely deleting the superseded file from disk upon successful commit.
- **Atomic Rollback:** If a database transaction fails during enrolment, any newly written image file on disk is immediately unlinked (`os.remove`), preventing orphaned files.

---

## 3. API Endpoints

| Method | Endpoint | Description | Auth Required |
|---|---|---|---|
| `POST` | `/api/v1/auth/login` | Authenticate and obtain access/refresh JWTs | None |
| `POST` | `/api/v1/auth/refresh` | Refresh access token | None |
| `GET` | `/api/v1/auth/me` | Current user profile | Bearer Token |
| `POST` | `/api/v1/academic/years` | Create academic year | Admin |
| `GET` | `/api/v1/academic/faculty/my-classes` | Faculty assigned classes | Faculty / Admin |
| `POST` | `/api/v1/students` | Register student profile | Admin |
| `POST` | `/api/v1/students/{id}/consent` | Log biometric consent metadata | Admin |
| `POST` | `/api/v1/biometrics/students/{student_id}/register-face` | Register reference face (multipart) | Admin / Assigned Faculty |
| `GET` | `/api/v1/biometrics/students/{student_id}/status` | Check student enrolment status | Admin / Assigned Faculty |

---

## 4. Running Tests

```bash
# Activate virtual environment
.\backend\venv\Scripts\Activate.ps1

# Run complete pytest test suite
pytest -v --tb=short
```

---

## 5. Docker Containerisation

The backend can be built and executed in an isolated, production-grade Linux container:

```bash
# 1. Build backend image
docker compose build

# 2. Start container in detached mode
docker compose up -d

# 3. View live logs
docker compose logs -f backend

# 4. Run automated test suite inside Docker
docker compose exec backend pytest -v --tb=short

# 5. Stop container
docker compose down
```

### Docker Storage & Security Architecture
- **Non-Root Execution:** Runs under `appuser` (UID 1000).
- **Persistent Database:** SQLite database persisted in named volume `reliattend_data` (`/app/data/reliattend.db`).
- **Persistent Biometrics:** Reference images persisted in named volume `reliattend_storage` (`/app/storage/reference_images/`).
- **Model Mount:** Pretrained ONNX weights (YuNet & SFace) mounted as **read-only bind mount** (`./backend/models:/app/models:ro`), keeping image sizes minimal and avoiding model duplication.

