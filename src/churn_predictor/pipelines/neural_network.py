from __future__ import annotations

import logging

import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from churn_predictor.data import (
    clean_telco_dataset,
    dataset_version_hash,
    download_dataset_if_needed,
    load_dataset,
    split_xy,
)
from churn_predictor.logging_utils import configure_logging, log_event
from churn_predictor.models.mlp import PyTorchMLPWrapper
from churn_predictor.pipelines.baselines import (
    DEFAULT_DATASET_PATH,
    DEFAULT_METRICS_PATH,
    DEFAULT_REPORTS_DIR,
    BusinessMetricConfig,
    build_preprocessor,
    evaluate_model,
    log_run_to_mlflow,
)

LOGGER = logging.getLogger(__name__)


def run_neural_network_pipeline(
    experiment_name: str = "telco-churn-baselines",
    random_state: int = 42,
    test_size: float = 0.2,
) -> pd.DataFrame:
    """Fluxo completo da etapa 2.

    dados -> pré-processamento -> treino PyTorch MLP -> avaliação -> tracking.
    """
    configure_logging()

    download_dataset_if_needed()
    raw_df = load_dataset()
    clean_df = clean_telco_dataset(raw_df)
    dataset_hash = dataset_version_hash(clean_df)

    x, y = split_xy(clean_df, target_col="Churn")

    # 1. Divisão inicial Treino e Teste
    x_train_full, x_test, y_train_full, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # 2. Construir preprocessor
    preprocessor = build_preprocessor(x_train_full)

    business_cfg = BusinessMetricConfig()
    mlflow.set_experiment(experiment_name)

    model_name = "pytorch_mlp"
    # Calcula input_dim usando um clone temporário para não consumir o preprocessor original
    from sklearn.base import clone

    _temp_prep = clone(preprocessor)
    input_dim = _temp_prep.fit_transform(x_train_full).shape[1]

    log_event(
        LOGGER,
        "model_training_started",
        event="model_training_started",
        model=model_name,
        step="train",
    )

    # 3. Inicializa o Wrapper
    mlp_model = PyTorchMLPWrapper(
        input_dim=input_dim,
        hidden_layers=[128, 64, 32],
        dropout_rate=0.3,
        learning_rate=1e-3,
        batch_size=64,
        epochs=150,
        patience=15,
    )

    # 4. Cria a Pipeline Completa
    mlp_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", mlp_model),
        ]
    )

    # 5. Treinamento
    mlp_pipeline.fit(x_train_full, y_train_full.values)

    # 6. Avaliação
    metrics = evaluate_model(mlp_pipeline, x_test, y_test, business_cfg)

    # 7. MLflow Tracking
    log_run_to_mlflow(
        model_name=model_name,
        model_pipeline=mlp_pipeline,
        metrics=metrics,
        x_test=x_test,
        y_test=y_test,
        dataset_hash=dataset_hash,
        dataset_path=DEFAULT_DATASET_PATH,
        business_cfg=business_cfg,
        output_dir=DEFAULT_REPORTS_DIR,
    )

    log_event(
        LOGGER,
        "model_training_completed",
        event="model_training_completed",
        model=model_name,
        step="evaluate",
    )

    # 8. Atualizar Tabela de Métricas
    new_row = {"model": model_name, "dataset_version_hash": dataset_hash, **metrics}

    if DEFAULT_METRICS_PATH.exists():
        metrics_df = pd.read_csv(DEFAULT_METRICS_PATH)
        # Remove a linha antiga da MLP se existir
        metrics_df = metrics_df[metrics_df["model"] != model_name]
        metrics_df = pd.concat([metrics_df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        metrics_df = pd.DataFrame([new_row])

    metrics_df = metrics_df.sort_values(by="auc_roc", ascending=False)
    DEFAULT_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(DEFAULT_METRICS_PATH, index=False)

    log_event(
        LOGGER,
        "neural_network_pipeline_completed",
        event="neural_network_pipeline_completed",
        model=model_name,
        step="done",
    )

    return metrics_df


if __name__ == "__main__":
    run_neural_network_pipeline()
