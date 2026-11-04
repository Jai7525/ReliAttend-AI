"""
ReliAttend AI — Health Check Automated Tests
Tests root health and versioned API health endpoints.
"""

from fastapi import status
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_root_health_returns_200():
    """Verify GET /health returns 200 OK and expected keys."""
    response = client.get("/health")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] == "healthy"
    assert "app_name" in data
    assert "version" in data
    assert "environment" in data


def test_api_v1_health_returns_200():
    """Verify GET /api/v1/health returns 200 OK and valid payload."""
    response = client.get("/api/v1/health")
    assert response.status_code == status.HTTP_200_OK
    payload = response.json()
    assert payload["success"] is True
    assert "message" in payload
    assert "data" in payload
    data = payload["data"]
    assert data["status"] == "healthy"
    assert "timestamp" in data

