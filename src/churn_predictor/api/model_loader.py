"""Carregamento do modelo serializado para inferência na API."""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path("models/best_model.joblib")

_cached_model: Pipeline | None = None


def load_model(path: Path = DEFAULT_MODEL_PATH) -> Pipeline:
    """Carrega o pipeline treinado do disco (com cache em memória).

    Args:
        path: Caminho para o arquivo .joblib do modelo.

    Returns:
        Pipeline sklearn treinado.

    Raises:
        FileNotFoundError: Se o arquivo do modelo não existir.
    """
    global _cached_model

    if _cached_model is not None:
        return _cached_model

    if not path.exists():
        msg = (
            f"Modelo não encontrado em '{path}'. "
            "Execute 'python -m churn_predictor.pipelines.baselines' primeiro."
        )
        raise FileNotFoundError(msg)

    _cached_model = joblib.load(path)
    LOGGER.info("model_loaded", extra={"path": str(path)})
    return _cached_model


def reset_model_cache() -> None:
    """Limpa o cache do modelo (útil para testes)."""
    global _cached_model
    _cached_model = None
