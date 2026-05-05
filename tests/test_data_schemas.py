"""Testes de schema pandera — validam DataFrames do dataset Telco Churn."""

from __future__ import annotations

import pandas as pd
import pytest
from pandera.errors import SchemaError

from churn_predictor.data_schemas import TELCO_FEATURES_SCHEMA, TELCO_RAW_SCHEMA


def _valid_row() -> dict:
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85,
        "Churn": "No",
    }


class TestTelcoRawSchema:
    """Schema do dataset bruto, com a coluna alvo Churn."""

    def test_valid_dataframe_passes(self):
        df = pd.DataFrame([_valid_row(), _valid_row()])
        validated = TELCO_RAW_SCHEMA.validate(df)
        assert len(validated) == 2

    def test_invalid_contract_fails(self):
        row = _valid_row()
        row["Contract"] = "Invalid contract"
        df = pd.DataFrame([row])
        with pytest.raises(SchemaError):
            TELCO_RAW_SCHEMA.validate(df)

    def test_negative_monthly_charges_fails(self):
        row = _valid_row()
        row["MonthlyCharges"] = -5.0
        df = pd.DataFrame([row])
        with pytest.raises(SchemaError):
            TELCO_RAW_SCHEMA.validate(df)

    def test_invalid_senior_citizen_fails(self):
        row = _valid_row()
        row["SeniorCitizen"] = 2
        df = pd.DataFrame([row])
        with pytest.raises(SchemaError):
            TELCO_RAW_SCHEMA.validate(df)


class TestTelcoFeaturesSchema:
    """Schema de features sem Churn, usado em batch input."""

    def test_features_schema_does_not_require_churn(self):
        row = _valid_row()
        del row["Churn"]
        df = pd.DataFrame([row])
        validated = TELCO_FEATURES_SCHEMA.validate(df)
        assert "Churn" not in validated.columns

    def test_features_schema_rejects_invalid_payment_method(self):
        row = _valid_row()
        del row["Churn"]
        row["PaymentMethod"] = "Bitcoin"
        df = pd.DataFrame([row])
        with pytest.raises(SchemaError):
            TELCO_FEATURES_SCHEMA.validate(df)
