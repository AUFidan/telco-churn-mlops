"""FastAPI model serving API."""

import os
from contextlib import asynccontextmanager

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Global model variable
model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    load_model()
    yield


app = FastAPI(
    title="Telco Churn Prediction API",
    description="API for predicting customer churn",
    version="1.0.0",
    lifespan=lifespan,
)


class CustomerFeatures(BaseModel):
    """Input features for a single customer."""

    gender: int
    SeniorCitizen: int
    Partner: int
    Dependents: int
    tenure: float
    PhoneService: int
    MultipleLines: int
    InternetService: int
    OnlineSecurity: int
    OnlineBackup: int
    DeviceProtection: int
    TechSupport: int
    StreamingTV: int
    StreamingMovies: int
    Contract: int
    PaperlessBilling: int
    PaymentMethod: int
    MonthlyCharges: float
    TotalCharges: float

    model_config = {
        "json_schema_extra": {
            "example": {
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
                "TotalCharges": 0.15,
            }
        }
    }


class PredictionResponse(BaseModel):
    """Prediction response."""

    churn_probability: float
    churn_prediction: int


class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""

    customers: list[CustomerFeatures]


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""

    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool


def load_model():
    """Load model from MLflow."""
    global model

    # Try to load from environment variable or default
    model_uri = os.getenv("MODEL_URI", None)
    model_name = os.getenv("MODEL_NAME", "telco-churn-ensemble")
    model_alias = os.getenv("MODEL_ALIAS", "staging")

    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(mlflow_uri)

    if model_uri:
        # Load from specific run
        logger.info(f"Loading model from URI: {model_uri}")
        model = mlflow.sklearn.load_model(model_uri)
    else:
        # Load from model registry using alias
        try:
            registry_uri = f"models:/{model_name}@{model_alias}"
            logger.info(f"Loading model from registry: {registry_uri}")
            model = mlflow.sklearn.load_model(registry_uri)
        except Exception as e:
            logger.warning(f"Could not load from registry: {e}", exc_info=True)
            logger.info("Model will need to be loaded manually")
            model = None

    return model


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(customer: CustomerFeatures):
    """Predict churn for a single customer."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Convert to DataFrame
    df = pd.DataFrame([customer.model_dump()])

    # Predict
    proba = model.predict_proba(df)[0, 1]
    prediction = int(proba >= 0.5)

    return PredictionResponse(
        churn_probability=float(proba),
        churn_prediction=prediction,
    )


@app.post("/predict_batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Predict churn for multiple customers."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Convert to DataFrame
    df = pd.DataFrame([c.model_dump() for c in request.customers])

    # Predict
    probas = model.predict_proba(df)[:, 1]
    predictions = (probas >= 0.5).astype(int)

    return BatchPredictionResponse(
        predictions=[
            PredictionResponse(
                churn_probability=float(p),
                churn_prediction=int(pred),
            )
            for p, pred in zip(probas, predictions)
        ]
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
