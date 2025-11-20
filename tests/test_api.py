from fastapi.testclient import TestClient
from app.main import app
import pytest

client = TestClient(app)


def test_root():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "LLM Inference API"
    assert "version" in response.json()


def test_health():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data


def test_inference_valid_request():
    """Test inference endpoint with valid request"""
    payload = {
        "prompt": "What is the meaning of life?",
        "max_tokens": 50,
        "temperature": 0.7,
    }
    response = client.post("/inference", json=payload)
    assert response.status_code == 200
    data = response.json()

    # Verify response schema
    assert "response" in data
    assert "tokens_used" in data
    assert "inference_time" in data
    assert "model" in data

    # Verify response types
    assert isinstance(data["response"], str)
    assert isinstance(data["tokens_used"], int)
    assert isinstance(data["inference_time"], float)
    assert data["model"] == "mock-llm-v1"


def test_inference_empty_prompt():
    """Test inference endpoint with empty prompt"""
    payload = {"prompt": "", "max_tokens": 50}
    response = client.post("/inference", json=payload)
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_inference_invalid_max_tokens():
    """Test inference endpoint with invalid max_tokens"""
    payload = {"prompt": "Test prompt", "max_tokens": -1}
    response = client.post("/inference", json=payload)
    assert response.status_code == 400


def test_inference_default_parameters():
    """Test inference endpoint with default parameters"""
    payload = {"prompt": "Test prompt with defaults"}
    response = client.post("/inference", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["tokens_used"] <= 100  # default max_tokens
