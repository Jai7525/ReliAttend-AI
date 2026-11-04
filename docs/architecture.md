# ReliAttend AI — Architecture Overview

## System Workflow & Core Principles
ReliAttend AI is a confidence-aware, multi-face recognition and reliability-based smart attendance management system.

1. **Frontend**: Responsive React web application for laptops and mobile browsers.
2. **Backend**: FastAPI modular architecture with layered separation:
   - `models/`: Database tables and relationships (SQLAlchemy)
   - `schemas/`: Request/Response validation contracts (Pydantic)
   - `routes/`: Thin API controllers
   - `services/`: Business logic and workflow enforcement
   - `ai/`: Independent computer vision pipeline components (detection, recognition, tracking, quality, PAD, reliability)
   - `core/`: Cross-cutting concerns (security, exceptions, logging)
3. **Storage & Privacy**: Protected local directories for biometrics (`storage/`), strictly uncommitted to source control.

