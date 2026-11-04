# ReliAttend AI — Backend Service

Modular FastAPI backend service for the ReliAttend AI Smart Attendance Management System.

## Architecture Structure
```
app/
├── models/       # Database entities (SQLAlchemy)
├── schemas/      # Request/Response schemas (Pydantic)
├── routes/       # API endpoints (FastAPI routers)
├── services/     # Business logic
├── ai/           # Computer vision & reliability engine
└── core/         # Security, exceptions, configuration
```

