"""Tests for FastAPI model serving API."""

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from src.api import main
from src.api.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_model():
    """Create a simple mock model for testing."""
    # Create a simple model that returns predictable results
    model = LogisticRegression()
    # Fit on dummy data
    X = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    y = np.array([0, 1])
    model.fit(X, y)
    return model


@pytest.fixture
def sample_customer():
    """Sample customer data for testing."""
    return {
        "gender": 1,
        "SeniorCitizen": 0,
        "Partner": 1,
        "Dependents": 0,
        "tenure": 0.5,
        "PhoneService": 1,
        "MultipleLines": 0,
        "InternetService": 1,
        "OnlineSecurity": 0,
        "OnlineBackup": 0,
        "DeviceProtection": 0,
        "TechSupport": 0,
        "StreamingTV": 0,
        "StreamingMovies": 0,
        "Contract": 0,
        "PaperlessBilling": 1,
        "PaymentMethod": 2,
        "MonthlyCharges": 0.2,
        "TotalCharges": -0.1,
    }


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_returns_200(self, client):
        """Test health endpoint returns 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_status(self, client):
        """Test health endpoint returns status field."""
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"

    def test_health_returns_model_loaded(self, client):
        """Test health endpoint returns model_loaded field."""
        response = client.get("/health")
        data = response.json()
        assert "model_loaded" in data


class TestPredictEndpoint:
    """Tests for /predict endpoint."""

    def test_predict_without_model_returns_503(self, client, sample_customer):
        """Test predict returns 503 when model not loaded."""
        # Ensure model is not loaded
        main.model = None
        response = client.post("/predict", json=sample_customer)
        assert response.status_code == 503

    def test_predict_with_model_returns_200(self, client, mock_model, sample_customer):
        """Test predict returns 200 when model is loaded."""
        main.model = mock_model
        response = client.post("/predict", json=sample_customer)
        assert response.status_code == 200

    def test_predict_returns_probability(self, client, mock_model, sample_customer):
        """Test predict returns churn probability."""
        main.model = mock_model
        response = client.post("/predict", json=sample_customer)
        data = response.json()
        assert "churn_probability" in data
        assert 0 <= data["churn_probability"] <= 1

    def test_predict_returns_prediction(self, client, mock_model, sample_customer):
        """Test predict returns churn prediction."""
        main.model = mock_model
        response = client.post("/predict", json=sample_customer)
        data = response.json()
        assert "churn_prediction" in data
        assert data["churn_prediction"] in [0, 1]

    def test_predict_invalid_input_returns_422(self, client, mock_model):
        """Test predict with invalid input returns 422."""
        main.model = mock_model
        response = client.post("/predict", json={"invalid": "data"})
        assert response.status_code == 422


class TestPredictBatchEndpoint:
    """Tests for /predict_batch endpoint."""

    def test_predict_batch_without_model_returns_503(self, client, sample_customer):
        """Test predict_batch returns 503 when model not loaded."""
        main.model = None
        response = client.post("/predict_batch", json={"customers": [sample_customer]})
        assert response.status_code == 503

    def test_predict_batch_with_model_returns_200(
        self, client, mock_model, sample_customer
    ):
        """Test predict_batch returns 200 when model is loaded."""
        main.model = mock_model
        response = client.post("/predict_batch", json={"customers": [sample_customer]})
        assert response.status_code == 200

    def test_predict_batch_returns_list(self, client, mock_model, sample_customer):
        """Test predict_batch returns list of predictions."""
        main.model = mock_model
        response = client.post(
            "/predict_batch", json={"customers": [sample_customer, sample_customer]}
        )
        data = response.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 2

    def test_predict_batch_each_has_probability_and_prediction(
        self, client, mock_model, sample_customer
    ):
        """Test each prediction in batch has required fields."""
        main.model = mock_model
        response = client.post("/predict_batch", json={"customers": [sample_customer]})
        data = response.json()
        for pred in data["predictions"]:
            assert "churn_probability" in pred
            assert "churn_prediction" in pred
