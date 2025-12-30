"""Pytest fixtures for tests."""

import pandas as pd
import pytest


@pytest.fixture
def sample_raw_data() -> pd.DataFrame:
    """Create sample raw data matching the Telco Churn schema."""
    # Need at least 2 samples per class for stratified split
    return pd.DataFrame(
        {
            "customerID": ["1-ABCD", "2-EFGH", "3-IJKL", "4-MNOP", "5-QRST", "6-UVWX"],
            "gender": ["Male", "Female", "Male", "Female", "Male", "Female"],
            "SeniorCitizen": [0, 1, 0, 1, 0, 0],
            "Partner": ["Yes", "No", "Yes", "No", "Yes", "No"],
            "Dependents": ["No", "No", "Yes", "No", "Yes", "No"],
            "tenure": [12, 0, 24, 36, 48, 6],
            "PhoneService": ["Yes", "Yes", "No", "Yes", "Yes", "No"],
            "MultipleLines": ["No", "No phone service", "No phone service", "Yes", "No", "No phone service"],
            "InternetService": ["DSL", "Fiber optic", "No", "DSL", "Fiber optic", "DSL"],
            "OnlineSecurity": ["Yes", "No", "No internet service", "Yes", "No", "Yes"],
            "OnlineBackup": ["No", "Yes", "No internet service", "No", "Yes", "No"],
            "DeviceProtection": ["No", "No", "No internet service", "Yes", "No", "No"],
            "TechSupport": ["Yes", "No", "No internet service", "Yes", "No", "Yes"],
            "StreamingTV": ["No", "Yes", "No internet service", "No", "Yes", "No"],
            "StreamingMovies": ["No", "No", "No internet service", "Yes", "No", "No"],
            "Contract": ["Month-to-month", "One year", "Two year", "Month-to-month", "One year", "Month-to-month"],
            "PaperlessBilling": ["Yes", "Yes", "No", "Yes", "No", "Yes"],
            "PaymentMethod": [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
                "Electronic check",
                "Mailed check",
            ],
            "MonthlyCharges": [29.85, 70.70, 20.05, 45.50, 89.90, 35.00],
            "TotalCharges": ["350.50", " ", "480.00", "1640.00", "4300.00", "210.00"],
            "Churn": ["No", "Yes", "No", "Yes", "No", "Yes"],  # 3 No, 3 Yes
        }
    )


@pytest.fixture
def sample_cleaned_data(sample_raw_data: pd.DataFrame) -> pd.DataFrame:
    """Create sample cleaned data (after cleaning step)."""
    from src.data.preprocessing import clean_data

    return clean_data(sample_raw_data)
