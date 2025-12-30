# Telco Customer Churn MLOps Project

MLOps project for predicting customer churn using the Telco Customer Churn dataset.

## Tech Stack

- **MLflow**: Experiment tracking & model registry
- **DVC**: Data versioning
- **Docker**: Containerized deployment
- **FastAPI**: Model serving API
- **uv**: Package management

## Setup

```bash
# Create and activate virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv sync --all-extras
```

## Project Structure

```
Project-1/
├── src/
│   ├── data/          # Data loading and preprocessing
│   ├── models/        # Model training
│   ├── api/           # FastAPI serving
│   └── utils/         # Utilities (logging, etc.)
├── tests/             # Unit tests
├── configs/           # Configuration files
├── data/              # Data directory (DVC tracked)
└── notebooks/         # Exploration notebooks
```
