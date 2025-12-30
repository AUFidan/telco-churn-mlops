"""Data preprocessing pipeline for Telco Customer Churn dataset."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_raw_data(path: Path | str = "data/raw/telco_churn.csv") -> pd.DataFrame:
    """Load raw data from CSV file."""
    logger.info(f"Loading raw data from {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw data."""
    logger.info("Cleaning data")
    df = df.copy()

    # Drop customerID - not useful for prediction
    df = df.drop(columns=["customerID"])

    # Fix TotalCharges - has whitespace values for new customers
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    # Fill NaN with 0 (new customers with tenure=0)
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    logger.info(f"Data cleaned: {len(df)} rows remaining")
    return df


def encode_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Encode categorical features."""
    logger.info("Encoding categorical features")
    df = df.copy()

    # Binary columns to encode as 0/1
    binary_cols = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]

    # Multi-class columns to encode
    multi_cols = [
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaymentMethod",
    ]

    encoders = {}

    # Encode binary columns
    for col in binary_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    # Encode multi-class columns
    for col in multi_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    # Encode target variable
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

    logger.info(f"Encoded {len(binary_cols) + len(multi_cols)} categorical columns")
    return df, encoders


def scale_features(
    df: pd.DataFrame, scaler: StandardScaler | None = None
) -> tuple[pd.DataFrame, StandardScaler]:
    """Scale numerical features."""
    logger.info("Scaling numerical features")
    df = df.copy()

    numerical_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

    if scaler is None:
        scaler = StandardScaler()
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    else:
        df[numerical_cols] = scaler.transform(df[numerical_cols])

    logger.info(f"Scaled {len(numerical_cols)} numerical columns")
    return df, scaler


def split_data(
    df: pd.DataFrame,
    target_col: str = "Churn",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into train and test sets."""
    logger.info(f"Splitting data with test_size={test_size}")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    logger.info(f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")
    return X_train, X_test, y_train, y_test


def preprocess_pipeline(
    raw_data_path: Path | str = "data/raw/telco_churn.csv",
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """
    Run full preprocessing pipeline.

    Returns:
        Dictionary containing:
        - X_train, X_test, y_train, y_test: Split data
        - scaler: Fitted StandardScaler
        - encoders: Dictionary of fitted LabelEncoders
        - feature_names: List of feature column names
    """
    logger.info("Starting preprocessing pipeline")

    # Load and clean
    df = load_raw_data(raw_data_path)
    df = clean_data(df)

    # Encode categorical features
    df, encoders = encode_features(df)

    # Split first (to avoid data leakage in scaling)
    X_train, X_test, y_train, y_test = split_data(
        df, test_size=test_size, random_state=random_state
    )

    # Scale numerical features (fit on train only)
    X_train, scaler = scale_features(X_train, scaler=None)
    X_test, _ = scale_features(X_test, scaler=scaler)

    logger.info("Preprocessing pipeline complete")

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "encoders": encoders,
        "feature_names": list(X_train.columns),
    }


if __name__ == "__main__":
    # Run preprocessing and print summary
    result = preprocess_pipeline()
    print(f"\nFeatures: {result['feature_names']}")
    print(f"X_train shape: {result['X_train'].shape}")
    print(f"X_test shape: {result['X_test'].shape}")
    print(f"y_train distribution:\n{result['y_train'].value_counts()}")
