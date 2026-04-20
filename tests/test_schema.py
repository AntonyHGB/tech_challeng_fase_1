"""Testes de schema — validam a entrada/saída Pydantic."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from churn_predictor.api.schemas import PredictionRequest, PredictionResponse


class TestPredictionRequest:
    """Testes para o schema de entrada."""

    def test_valid_payload(self, valid_payload: dict):
        """Aceita um payload válido."""
        req = PredictionRequest(**valid_payload)
        assert req.gender == "Female"
        assert req.tenure == 1
        assert req.monthly_charges == 29.85

    def test_missing_required_field(self, valid_payload: dict):
        """Rejeita payload com campo obrigatório ausente."""
        del valid_payload["gender"]
        with pytest.raises(ValidationError):
            PredictionRequest(**valid_payload)

    def test_invalid_senior_citizen(self, valid_payload: dict):
        """Rejeita SeniorCitizen fora do range [0, 1]."""
        valid_payload["SeniorCitizen"] = 5
        with pytest.raises(ValidationError):
            PredictionRequest(**valid_payload)

    def test_negative_charges(self, valid_payload: dict):
        """Rejeita MonthlyCharges negativo."""
        valid_payload["MonthlyCharges"] = -10.0
        with pytest.raises(ValidationError):
            PredictionRequest(**valid_payload)


class TestPredictionResponse:
    """Testes para o schema de saída."""

    def test_valid_response(self):
        """Constrói uma resposta válida."""
        resp = PredictionResponse(
            churn_probability=0.72,
            churn_prediction=1,
            model_used="logistic_regression",
        )
        assert resp.churn_probability == 0.72
        assert resp.churn_prediction == 1

    def test_invalid_probability_range(self):
        """Rejeita probabilidade fora de [0, 1]."""
        with pytest.raises(ValidationError):
            PredictionResponse(
                churn_probability=1.5,
                churn_prediction=1,
                model_used="logistic_regression",
            )
