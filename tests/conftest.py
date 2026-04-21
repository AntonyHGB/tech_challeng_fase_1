"""Fixtures compartilhadas para os testes."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from churn_predictor.api.app import app

if TYPE_CHECKING:
    from fastapi.testclient import TestClient


@pytest.fixture
def client() -> TestClient:
    """Cria um TestClient com um modelo mockado."""
    from unittest.mock import patch

    import numpy as np
    from fastapi.testclient import TestClient as _TestClient

    # Mock do modelo para não depender de arquivo .joblib nos testes.
    mock_pipeline = MagicMock()
    mock_pipeline.predict_proba.return_value = np.array([[0.3, 0.7]])

    with patch("churn_predictor.api.model_loader._cached_model", new=mock_pipeline):
        yield _TestClient(app)


@pytest.fixture
def valid_payload() -> dict:
    """Payload válido para o endpoint /predict."""
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
    }
