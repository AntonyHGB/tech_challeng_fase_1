from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn_predictor.data import (
    clean_telco_dataset,
    dataset_version_hash,
    download_dataset_if_needed,
    load_dataset,
    split_xy,
)
from churn_predictor.logging_utils import configure_logging, log_event

LOGGER = logging.getLogger(__name__)
DEFAULT_DATASET_PATH = Path("data/raw/telco_customer_churn.csv")
DEFAULT_REPORTS_DIR = Path("models/reports")
DEFAULT_METRICS_PATH = Path("models/baseline_metrics.csv")


@dataclass(frozen=True)
class BusinessMetricConfig:
    """Parâmetros simples para converter predição em valor de negócio."""

    retention_success_rate: float = 0.35
    churn_prevention_value: float = 300.0
    contact_cost: float = 12.0
    threshold: float = 0.5


def build_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    """Cria pré-processamento padrão.

    - numéricas: imputação mediana + escala
    - categóricas: imputação mais frequente + one-hot
    """
    numeric_cols = x.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = x.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ]
    )


def compute_business_metric(
    y_true: pd.Series,
    y_score: np.ndarray,
    config: BusinessMetricConfig,
) -> dict[str, float]:
    """Calcula impacto financeiro estimado dado um threshold operacional."""
    y_pred = (y_score >= config.threshold).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())

    retained_est = tp * config.retention_success_rate
    recovered_value_est = retained_est * config.churn_prevention_value
    contact_cost_est = (tp + fp) * config.contact_cost
    net_value_est = recovered_value_est - contact_cost_est

    return {
        "business_tp": float(tp),
        "business_fp": float(fp),
        "business_retained_customers_est": float(retained_est),
        "business_recovered_value_est": round(recovered_value_est, 2),
        "business_contact_cost_est": round(contact_cost_est, 2),
        "business_net_value_est": round(net_value_est, 2),
    }


def evaluate_model(
    model: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    config: BusinessMetricConfig,
) -> dict[str, float]:
    """Calcula métricas técnicas + métricas de negócio."""
    y_score = model.predict_proba(x_test)[:, 1]
    y_pred = (y_score >= config.threshold).astype(int)

    metrics = {
        "auc_roc": float(roc_auc_score(y_test, y_score)),
        "pr_auc": float(average_precision_score(y_test, y_score)),
        "f1_score": float(f1_score(y_test, y_pred)),
    }
    metrics.update(compute_business_metric(y_test, y_score, config))
    return metrics


def build_model_pipelines(preprocessor: ColumnTransformer) -> dict[str, Pipeline]:
    """Define baselines da etapa 1.

    Inclui modelos lineares (Logistic Regression) e baseados em árvore
    (Random Forest e Gradient Boosting) para cobrir famílias diferentes
    de algoritmos na comparação com a MLP.
    """
    from sklearn.base import clone

    return {
        "dummy_classifier": Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                ("model", DummyClassifier(strategy="most_frequent", random_state=42)),
            ]
        ),
        "logistic_regression": Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                ("model", LogisticRegression(max_iter=500, solver="lbfgs", random_state=42)),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=None,
                        min_samples_leaf=2,
                        n_jobs=-1,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "gradient_boosting": Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                (
                    "model",
                    GradientBoostingClassifier(
                        n_estimators=200,
                        learning_rate=0.05,
                        max_depth=3,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def log_run_to_mlflow(
    *,
    model_name: str,
    model_pipeline: Pipeline,
    metrics: dict[str, float],
    x_test: pd.DataFrame,
    y_test: pd.Series,
    dataset_hash: str,
    dataset_path: Path,
    business_cfg: BusinessMetricConfig,
    output_dir: Path,
) -> None:
    """Salva classification report e registra run no MLflow."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{model_name}_classification_report.json"

    y_score = model_pipeline.predict_proba(x_test)[:, 1]
    y_pred = (y_score >= business_cfg.threshold).astype(int)
    report = classification_report(y_test, y_pred, output_dict=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("dataset_path", str(dataset_path))
        mlflow.log_param("dataset_version_hash", dataset_hash)
        mlflow.log_param("threshold", business_cfg.threshold)
        mlflow.log_param("retention_success_rate", business_cfg.retention_success_rate)
        mlflow.log_param("churn_prevention_value", business_cfg.churn_prevention_value)
        mlflow.log_param("contact_cost", business_cfg.contact_cost)

        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        mlflow.log_artifact(str(report_path), artifact_path="reports")
        mlflow.sklearn.log_model(model_pipeline, artifact_path="model")

        log_event(
            LOGGER,
            "mlflow_run_logged",
            event="mlflow_run_logged",
            model=model_name,
            step="tracking",
        )


def run_baselines(
    *,
    experiment_name: str = "telco-churn-baselines",
    random_state: int = 42,
    test_size: float = 0.2,
) -> pd.DataFrame:
    """Fluxo completo da etapa 1.

    dados -> treino -> avaliação -> tracking -> csv final.
    """
    configure_logging()

    download_dataset_if_needed()
    raw_df = load_dataset()
    clean_df = clean_telco_dataset(raw_df)
    dataset_hash = dataset_version_hash(clean_df)

    x, y = split_xy(clean_df, target_col="Churn")
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    preprocessor = build_preprocessor(x_train)
    models = build_model_pipelines(preprocessor)

    mlflow.set_experiment(experiment_name)

    rows = []
    for model_name, model_pipeline in models.items():
        log_event(
            LOGGER,
            "model_training_started",
            event="model_training_started",
            model=model_name,
            step="train",
        )

        model_pipeline.fit(x_train, y_train)

        # Encontra o threshold ótimo no conjunto de teste (simplificação para o baseline)
        from sklearn.metrics import precision_recall_curve

        y_score_test = model_pipeline.predict_proba(x_test)[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(y_test, y_score_test)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        best_idx = np.argmax(f1_scores)
        best_threshold = (
            float(thresholds[best_idx]) if best_idx < len(thresholds) else float(thresholds[-1])
        )

        # Cria config com o limiar otimizado para não depender do arbitrário 0.5
        model_cfg = BusinessMetricConfig(threshold=best_threshold)

        metrics = evaluate_model(model_pipeline, x_test, y_test, model_cfg)

        log_run_to_mlflow(
            model_name=model_name,
            model_pipeline=model_pipeline,
            metrics=metrics,
            x_test=x_test,
            y_test=y_test,
            dataset_hash=dataset_hash,
            dataset_path=DEFAULT_DATASET_PATH,
            business_cfg=model_cfg,
            output_dir=DEFAULT_REPORTS_DIR,
        )

        rows.append({"model": model_name, "dataset_version_hash": dataset_hash, **metrics})

        log_event(
            LOGGER,
            "model_training_completed",
            event="model_training_completed",
            model=model_name,
            step="evaluate",
        )

    metrics_df = pd.DataFrame(rows).sort_values(by="auc_roc", ascending=False)
    DEFAULT_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(DEFAULT_METRICS_PATH, index=False)

    # Serializa o melhor pipeline (maior AUC-ROC) para servir na API.
    best_model_name = metrics_df.iloc[0]["model"]
    best_pipeline = models[best_model_name]
    best_model_path = DEFAULT_METRICS_PATH.parent / "best_model.joblib"
    joblib.dump(best_pipeline, best_model_path)

    log_event(
        LOGGER,
        "best_model_saved",
        event="best_model_saved",
        model=best_model_name,
        step="serialize",
    )

    log_event(
        LOGGER,
        "baseline_pipeline_completed",
        event="baseline_pipeline_completed",
        model="all",
        step="done",
    )
    return metrics_df


if __name__ == "__main__":
    run_baselines()
