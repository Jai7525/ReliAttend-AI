# ReliAttend AI

**Confidence-Aware Multi-Face Recognition and Reliability-Based Smart Attendance Management System**

## Quick Start (Backend)

### 1. Activate Virtual Environment
```powershell
cd backend
.\venv\Scripts\Activate.ps1
```

### 2. Run the Development Server
```powershell
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

### 3. Verify Health & OpenAPI Docs
- **Root Health:** [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)
- **API v1 Health:** [http://127.0.0.1:8000/api/v1/health](http://127.0.0.1:8000/api/v1/health)
- **Interactive Swagger Docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Interactive ReDoc:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

### 4. Run Automated Tests
```powershell
cd backend
.\venv\Scripts\pytest
```

### 5. Postman Testing
Import `backend/postman/ReliAttend_API.json` into Postman and execute the **00 - Health Check** folder.

