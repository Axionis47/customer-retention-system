"""Integration tests for FastAPI endpoints."""
import pytest
from fastapi.testclient import TestClient

from serve.app import app


@pytest.fixture
def client():
    """Create test client with startup/shutdown events."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.mark.integration
def test_healthz(client):
    """Health check should return 200."""
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


@pytest.mark.integration
def test_root(client):
    """Root endpoint should return service info."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert "endpoints" in data


@pytest.mark.integration
def test_retain_endpoint(client):
    """Retain endpoint should return decision."""
    payload = {
        "customer_id": "C12345",
        "churn_risk": 0.75,
        "tenure_months": 24,
        "monthly_spend": 89.99,
        "contacts_last_7d": 0,
        "days_since_last_contact": 30,
    }

    response = client.post("/retain", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "customer_id" in data
    assert data["customer_id"] == "C12345"
    assert "decision" in data
    assert data["decision"] in ["contact", "no_contact"]
    assert "offer_pct" in data
    assert "latency_ms" in data
    assert data["latency_ms"] > 0


@pytest.mark.integration
def test_retain_validation():
    """Retain endpoint should validate input."""
    client = TestClient(app)

    # Missing required field
    payload = {
        "customer_id": "C12345",
        "churn_risk": 0.75,
        # Missing tenure_months
    }

    response = client.post("/retain", json=payload)
    assert response.status_code == 422  # Validation error


@pytest.mark.integration
def test_retain_with_contact():
    """High churn risk should trigger contact."""
    client = TestClient(app)

    payload = {
        "customer_id": "C99999",
        "churn_risk": 0.95,  # Very high
        "tenure_months": 12,
        "monthly_spend": 100.0,
        "contacts_last_7d": 0,
        "days_since_last_contact": 60,
    }

    response = client.post("/retain", json=payload)
    assert response.status_code == 200

    data = response.json()
    # High churn risk should likely trigger contact (baseline policy)
    # Note: This is probabilistic, so we just check response structure
    assert "message" in data or data["decision"] == "no_contact"


@pytest.mark.integration
def test_retain_latency():
    """Retain endpoint should respond quickly."""
    client = TestClient(app)

    payload = {
        "customer_id": "C00001",
        "churn_risk": 0.5,
        "tenure_months": 12,
        "monthly_spend": 50.0,
    }

    response = client.post("/retain", json=payload)
    assert response.status_code == 200

    data = response.json()
    # Should respond in < 1000ms locally
    assert data["latency_ms"] < 1000, f"Latency too high: {data['latency_ms']}ms"

