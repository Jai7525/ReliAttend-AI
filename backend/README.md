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
│   │   ├── student.py    # Students, ConsentRecords, FaceReferences (1-to-many BLOB embeddings)
│   │   ├── user.py       # Users (Admin & Faculty RBAC)
│   │   └── __init__.py
│   ├── routes/           # FastAPI REST API route endpoints
│   │   ├── academic.py
│   │   ├── auth.py
│   │   ├── biometrics.py # Multi-reference enrollment & status inquiries
│   │   ├── deps.py       # Authentication & scoped RBAC guards
│   │   ├── health.py
│   │   ├── students.py
│   │   └── __init__.py
│   ├── schemas/          # Pydantic v2 data validation schemas
│   │   ├── academic.py
│   │   ├── auth.py
│   │   ├── biometrics.py # Privacy-preserving responses (zero path/embedding leaks)
│   │   ├── student.py
│   │   └── __init__.py
│   ├── services/         # Business & domain services
│   │   ├── academic_service.py
│   │   ├── auth_service.py
│   │   ├── biometric_service.py # Multi-reference orchestration, rollback safety, self-healing promotion
│   │   ├── student_service.py
│   │   └── __init__.py
│   ├── config.py         # Application settings (Pydantic BaseSettings)
│   ├── database.py       # SQLAlchemy engine & session factory
│   └── main.py           # FastAPI entrypoint, middleware, lifespan
├── models/               # Local ONNX weights (gitignored: YuNet & SFace)
├── postman/              # Postman collection (ReliAttend_API.json)
├── tests/                # Automated pytest test suites (58 tests across 8 modules)
└── requirements.txt
```

---

## 2. Multi-Reference Biometric Architecture

### 2.1 One-to-Many Reference Enrolment
The biometric storage architecture links each student to **up to 5 active reference images** (`FaceReference` table with indexed `student_id`).
- **Minimum for Enrollment:** $\ge 1$ active reference marks `is_enrolled = True`.
- **Recommended Count:** 3 to 5 images covering natural pose variations.
- **Maximum Active Cap:** Exactly 5 active references per student. Any 6th upload attempt returns `409 Conflict`.

### 2.2 Supported Natural Poses (`pose_label`)
Allowed pose labels (configurable via `BIOMETRIC_ALLOWED_POSES`):
1. `frontal` (default / primary view)
2. `slight_left` (~15° natural yaw left)
3. `slight_right` (~15° natural yaw right)
4. `slight_up` (~10° natural pitch up)
5. `slight_down` (~10° natural pitch down)
6. `custom`

*Note: `pose_label` is user/operator provided angle metadata, not unvalidated automatic 3D head pose estimation.*

### 2.3 Primary Reference Consistency & Self-Healing Deletion
- Each student has **at most one primary reference** (`is_primary = True`).
- Registering a new primary reference automatically clears the primary flag on older active references.
- When the primary reference is deleted:
  - The system **automatically self-heals** by promoting the newest remaining active reference to primary.
- When all active references are deleted:
  - The student's `registration_status` automatically reverts to `pending` and `is_enrolled` becomes `False`.

### 2.4 Experimental Recognition Matching Pipeline
When matching a live camera face $v_{\text{live}}$ against a student's $K$ reference embeddings $\{r_1, \dots, r_K\}$:
- Cosine similarity: $s_i = v_{\text{live}} \cdot r_i$.
- **Best Match:** $S_{\text{final}} = \max(s_1, \dots, s_K)$.
- *Experimental Threshold:* $0.65$ (Configurable in `config.py` as `BIOMETRIC_EXPERIMENTAL_MATCH_THRESHOLD`; treated as experimental pending formal field validation).

---

## 3. API Endpoints

| Method | Endpoint | Description | Auth Required |
|---|---|---|---|
| `POST` | `/api/v1/auth/login` | Authenticate and obtain JWTs | None |
| `POST` | `/api/v1/academic/years` | Create academic year | Admin |
| `POST` | `/api/v1/students` | Register student profile | Admin |
| `POST` | `/api/v1/students/{id}/consent` | Log active biometric consent | Admin |
| `POST` | `/api/v1/biometrics/students/{student_id}/references` | **Add Incremental Reference** (`pose_label`, `is_primary`) | Admin / Assigned Faculty |
| `GET` | `/api/v1/biometrics/students/{student_id}/references` | **List Active References** (Zero-leak metadata) | Admin / Assigned Faculty |
| `DELETE` | `/api/v1/biometrics/students/{student_id}/references/{ref_id}` | **Delete Reference** (Promotes new primary, unlinks file) | Admin / Assigned Faculty |
| `POST` | `/api/v1/biometrics/students/{student_id}/register-face` | Register primary frontal reference (Backward-compatible) | Admin / Assigned Faculty |
| `GET` | `/api/v1/biometrics/students/{student_id}/status` | Get biometric enrollment status & active pose list | Admin / Assigned Faculty |

---

## 4. Local Real-Photo Testing Guide (Zero Git Leaks)

You can test the multi-reference system on your own computer with your own face without uploading any photos to GitHub or external servers.

### Step 1: Capture 3–5 Test Photos
Take 3 to 5 clear photos of yourself on your laptop webcam:
1. `photo_frontal.jpg`: Front-facing, neutral expression, clear lighting.
2. `photo_left.jpg`: Head angled ~15° to your left.
3. `photo_right.jpg`: Head angled ~15° to your right.
4. `photo_smile.jpg`: Slight smile or minor angle variation.

*Store these files locally on your machine outside the repository (e.g., `C:\temp\my_photos\`) or inside `test_data/my_photos/` (which is already gitignored).*

### Step 2: Ingest via Postman or Curl
1. **Login:** Call `POST /api/v1/auth/login` as `admin` (`Admin@123`) $\rightarrow$ Postman saves `{{access_token}}`.
2. **Create Student Profile:** Call `POST /api/v1/students` with your student ID (e.g., `STU_MYSELF_01`) and class section ID `1`.
3. **Record Consent:** Call `POST /api/v1/students/{{student_pk}}/consent` (BR-018 requirement).
4. **Upload Frontal Reference:**
   - Call `POST /api/v1/biometrics/students/STU_MYSELF_01/references`
   - Form-data: `file` = select `photo_frontal.jpg`, `pose_label` = `frontal`, `is_primary` = `true`.
5. **Upload Left Angle:**
   - Call `POST /api/v1/biometrics/students/STU_MYSELF_01/references`
   - Form-data: `file` = select `photo_left.jpg`, `pose_label` = `slight_left`, `is_primary` = `false`.
6. **Upload Right Angle:**
   - Call `POST /api/v1/biometrics/students/STU_MYSELF_01/references`
   - Form-data: `file` = select `photo_right.jpg`, `pose_label` = `slight_right`, `is_primary` = `false`.
7. **Verify List:**
   - Call `GET /api/v1/biometrics/students/STU_MYSELF_01/references`
   - Inspect response: 3 references listed with pose labels, quality scores, and zero leaked server paths.
8. **Check Status:**
   - Call `GET /api/v1/biometrics/students/STU_MYSELF_01/status`
   - Response shows `is_enrolled: true`, `active_reference_count: 3`, `active_poses: ["frontal", "slight_left", "slight_right"]`.
9. **Verify Privacy:**
   - Run `git status` in your terminal. Git will remain completely clean: no photos, embeddings, or DB files are tracked.

---

## 5. Running Automated Tests

```bash
# Activate virtual environment
.\backend\venv\Scripts\Activate.ps1

# Run complete pytest test suite (58 tests)
pytest -v --tb=short
```
