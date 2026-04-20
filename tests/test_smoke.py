"""Smoke tests — verificam que os módulos principais importam corretamente."""

from __future__ import annotations


def test_import_data_module():
    """Verifica que o módulo de dados importa sem erros."""
    from churn_predictor.data import (
        clean_telco_dataset,
        download_dataset_if_needed,
        load_dataset,
        split_xy,
    )

    assert callable(clean_telco_dataset)
    assert callable(download_dataset_if_needed)
    assert callable(load_dataset)
    assert callable(split_xy)


def test_import_baselines_module():
    """Verifica que o módulo de baselines importa sem erros."""
    from churn_predictor.pipelines.baselines import (
        build_preprocessor,
        evaluate_model,
        run_baselines,
    )

    assert callable(build_preprocessor)
    assert callable(evaluate_model)
    assert callable(run_baselines)


def test_import_mlp_module():
    """Verifica que o módulo da rede neural importa sem erros."""
    from churn_predictor.models.mlp import ChurnMLP, PyTorchMLPWrapper

    assert ChurnMLP is not None
    assert PyTorchMLPWrapper is not None


def test_import_api_app():
    """Verifica que a aplicação FastAPI instancia sem erros."""
    from churn_predictor.api.app import app

    assert app is not None
    assert app.title == "Churn Predictor API"
