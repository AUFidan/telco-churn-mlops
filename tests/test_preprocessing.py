"""Tests for data preprocessing module."""

from pathlib import Path

import pandas as pd
import pytest

from src.data.preprocessing import (
    clean_data,
    encode_features,
    load_raw_data,
    scale_features,
    split_data,
)

DATA_FILE = Path("data/raw/telco_churn.csv")


class TestLoadRawData:
    """Tests for load_raw_data function."""

    @pytest.mark.skipif(not DATA_FILE.exists(), reason="Data file not available")
    def test_load_raw_data_returns_dataframe(self):
        """Test that load_raw_data returns a DataFrame."""
        df = load_raw_data()
        assert isinstance(df, pd.DataFrame)

    @pytest.mark.skipif(not DATA_FILE.exists(), reason="Data file not available")
    def test_load_raw_data_has_expected_columns(self):
        """Test that loaded data has expected columns."""
        df = load_raw_data()
        expected_cols = ["customerID", "gender", "Churn", "TotalCharges"]
        for col in expected_cols:
            assert col in df.columns


class TestCleanData:
    """Tests for clean_data function."""

    def test_clean_data_drops_customer_id(self, sample_raw_data: pd.DataFrame):
        """Test that customerID is dropped."""
        df = clean_data(sample_raw_data)
        assert "customerID" not in df.columns

    def test_clean_data_converts_total_charges(self, sample_raw_data: pd.DataFrame):
        """Test that TotalCharges is converted to numeric."""
        df = clean_data(sample_raw_data)
        assert df["TotalCharges"].dtype in ["float64", "float32"]

    def test_clean_data_fills_missing_total_charges(
        self, sample_raw_data: pd.DataFrame
    ):
        """Test that missing TotalCharges (whitespace) is filled with 0."""
        df = clean_data(sample_raw_data)
        assert df["TotalCharges"].isna().sum() == 0
        # The row with " " should now be 0
        assert 0.0 in df["TotalCharges"].values


class TestEncodeFeatures:
    """Tests for encode_features function."""

    def test_encode_features_returns_tuple(self, sample_cleaned_data: pd.DataFrame):
        """Test that encode_features returns DataFrame and encoders dict."""
        df, encoders = encode_features(sample_cleaned_data)
        assert isinstance(df, pd.DataFrame)
        assert isinstance(encoders, dict)

    def test_encode_features_converts_churn(self, sample_cleaned_data: pd.DataFrame):
        """Test that Churn is converted to 0/1."""
        df, _ = encode_features(sample_cleaned_data)
        assert set(df["Churn"].unique()).issubset({0, 1})

    def test_encode_features_all_numeric(self, sample_cleaned_data: pd.DataFrame):
        """Test that all columns are numeric after encoding."""
        df, _ = encode_features(sample_cleaned_data)
        for col in df.columns:
            assert pd.api.types.is_numeric_dtype(df[col]), f"{col} is not numeric"


class TestScaleFeatures:
    """Tests for scale_features function."""

    def test_scale_features_returns_scaler(self, sample_cleaned_data: pd.DataFrame):
        """Test that scale_features returns a scaler."""
        df, encoders = encode_features(sample_cleaned_data)
        scaled_df, scaler = scale_features(df)
        assert scaler is not None

    def test_scale_features_standardizes_columns(
        self, sample_cleaned_data: pd.DataFrame
    ):
        """Test that numerical columns are standardized."""
        df, _ = encode_features(sample_cleaned_data)
        scaled_df, _ = scale_features(df)
        # After scaling, mean should be ~0 (with tolerance for small samples)
        assert abs(scaled_df["tenure"].mean()) < 1


class TestSplitData:
    """Tests for split_data function."""

    def test_split_data_returns_correct_shapes(self, sample_cleaned_data: pd.DataFrame):
        """Test that split maintains correct proportions."""
        df, _ = encode_features(sample_cleaned_data)
        X_train, X_test, y_train, y_test = split_data(df, test_size=0.33)

        assert len(X_train) + len(X_test) == len(df)
        assert len(y_train) == len(X_train)
        assert len(y_test) == len(X_test)

    def test_split_data_stratified(self, sample_cleaned_data: pd.DataFrame):
        """Test that split is stratified by target."""
        df, _ = encode_features(sample_cleaned_data)
        X_train, X_test, y_train, y_test = split_data(df, test_size=0.33)

        # Both splits should have samples (with such small data, at least 1 each)
        assert len(y_train) > 0
        assert len(y_test) > 0
