"""Testes da API — verificam os endpoints /health e /predict."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi.testclient import TestClient


class TestHealthEndpoint:
    """Testes para o endpoint GET /health."""

    def test_health_returns_200(self, client: TestClient):
        """Health check retorna status 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_ok(self, client: TestClient):
        """Health check retorna body correto."""
        response = client.get("/health")
        assert response.json() == {"status": "ok"}


class TestPredictEndpoint:
    """Testes para o endpoint POST /predict."""

    def test_predict_returns_200(self, client: TestClient, valid_payload: dict):
        """Predição com payload válido retorna status 200."""
        response = client.post("/predict", json=valid_payload)
        assert response.status_code == 200

    def test_predict_response_schema(self, client: TestClient, valid_payload: dict):
        """Resposta contém os campos esperados."""
        response = client.post("/predict", json=valid_payload)
        data = response.json()
        assert "churn_probability" in data
        assert "churn_prediction" in data
        assert "model_used" in data

    def test_predict_probability_range(self, client: TestClient, valid_payload: dict):
        """Probabilidade de churn está entre 0 e 1."""
        response = client.post("/predict", json=valid_payload)
        data = response.json()
        assert 0.0 <= data["churn_probability"] <= 1.0

    def test_predict_invalid_payload_returns_422(self, client: TestClient):
        """Payload inválido retorna status 422."""
        response = client.post("/predict", json={"invalid": "data"})
        assert response.status_code == 422

    def test_predict_empty_payload_returns_422(self, client: TestClient):
        """Payload vazio retorna status 422."""
        response = client.post("/predict", json={})
        assert response.status_code == 422
