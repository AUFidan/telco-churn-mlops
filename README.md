# Telco Customer Churn MLOps Project

End-to-end MLOps pipeline for predicting customer churn using the Telco Customer Churn dataset.

## Tech Stack

| Component | Tool |
|-----------|------|
| Experiment Tracking | MLflow |
| Data Versioning | DVC |
| Hyperparameter Tuning | Optuna |
| Model Serving | FastAPI |
| Containerization | Docker |
| Package Management | uv |
| Testing | pytest |
| CI/CD | GitHub Actions |

## Models

- Logistic Regression (baseline)
- XGBoost
- LightGBM
- CatBoost
- **Ensemble**: StackingClassifier with LogisticRegression meta-learner

## Project Structure

```
Project-1/
├── .github/workflows/     # CI/CD pipelines
├── configs/               # Configuration files
├── data/
│   ├── raw/              # Raw data (DVC tracked)
│   └── processed/        # Processed data
├── src/
│   ├── api/              # FastAPI serving
│   ├── data/             # Data preprocessing
│   ├── models/           # Training scripts
│   └── utils/            # Utilities (logging)
├── tests/                # Unit tests
├── docker-compose.yml    # Infrastructure
├── Dockerfile.api        # API container
└── Dockerfile.mlflow     # MLflow container
```

## Setup

### Prerequisites

- Python 3.10+
- Docker & Docker Compose
- uv (Python package manager)

### Installation

```bash
# Clone repository
git clone <repo-url>
cd Project-1

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate
uv sync --all-extras

# Pull data from DVC
dvc pull
```

### Start Infrastructure

```bash
# Start MLflow, PostgreSQL, MinIO
docker-compose up -d postgres minio minio-setup mlflow

# Verify MLflow is running
open http://localhost:5000
```

## Usage

### Train a Model

```bash
# Train single model
uv run python -m src.models.train --model logistic_regression

# Train with model registration
uv run python -m src.models.train --model xgboost --register

# Train ensemble
uv run python -m src.models.train --model ensemble --register
```

### Run API

```bash
# Local development
uv run uvicorn src.api.main:app --reload

# Docker
docker-compose up -d api
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/predict` | POST | Single prediction |
| `/predict_batch` | POST | Batch predictions |

Example request:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
    "TotalCharges": -0.1
  }'
```

### Run Tests

```bash
# All tests
uv run pytest tests/ -v

# With coverage
uv run pytest tests/ -v --cov=src
```

### Linting

```bash
uv run ruff check .
uv run ruff format .
```

## Configuration

Edit `configs/config.yaml` to modify:

- Data paths and split ratios
- MLflow tracking URI
- Optuna tuning parameters (n_trials, cv_folds)
- Model hyperparameter search spaces

## MLflow UI

Access at http://localhost:5000 after starting infrastructure.

Features:
- View experiment runs
- Compare metrics across models
- Model registry with staging/production stages
- Artifact storage (models, plots)

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Training  │────▶│   MLflow    │────▶│   Model     │
│   Script    │     │   Server    │     │   Registry  │
└─────────────┘     └─────────────┘     └─────────────┘
                           │
                    ┌──────┴──────┐
                    │             │
              ┌─────▼─────┐ ┌─────▼─────┐
              │ PostgreSQL│ │   MinIO   │
              │ (metadata)│ │(artifacts)│
              └───────────┘ └───────────┘

┌─────────────┐     ┌─────────────┐
│   FastAPI   │────▶│   Model     │
│     API     │     │  (loaded)   │
└─────────────┘     └─────────────┘
```
