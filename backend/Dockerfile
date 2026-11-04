# ==============================================================================
# ReliAttend AI — Backend Production Dockerfile
# Multi-stage/Slim Linux container running FastAPI with Uvicorn.
# Includes OpenCV headless runtimes, non-root user execution, and health checks.
# ==============================================================================

FROM python:3.13-slim-bookworm

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install minimal OS dependencies for OpenCV headless (GLib, GOMP) and healthchecks (curl)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create unprivileged system user and group for container security
RUN groupadd -g 1000 appgroup && \
    useradd -u 1000 -g appgroup -m -s /bin/bash appuser

# Upgrade pip and install Python package dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code, tests, and configuration
COPY app /app/app
COPY tests /app/tests

# Create dedicated persistent directories for SQLite database, storage, and models
RUN mkdir -p /app/data /app/storage/reference_images /app/models && \
    chown -R appuser:appgroup /app/data /app/storage /app/models /app

# Switch to unprivileged user
USER appuser

# Expose backend API port
EXPOSE 8000

# Container healthcheck querying API v1 health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Default startup command running Uvicorn ASGI server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

