from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
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


@dataclass(frozen=True)
class BusinessMetricConfig:
    retention_success_rate: float = 0.35
    churn_prevention_value: float = 300.0
    contact_cost: float = 12.0
    threshold: float = 0.5


def build_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    # separa features numéricas e categóricas para tratamento dedicado
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
) -> Dict[str, float]:
    # aplica threshold operacional para estimar impacto financeiro da campanha
    y_pred = (y_score >= config.threshold).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())

    estimated_retained_customers = tp * config.retention_success_rate
    estimated_recovered_value = estimated_retained_customers * config.churn_prevention_value
    estimated_contact_cost = (tp + fp) * config.contact_cost
    estimated_net_value = estimated_recovered_value - estimated_contact_cost

    return {
        "business_tp": float(tp),
        "business_fp": float(fp),
        "business_retained_customers_est": float(estimated_retained_customers),
        "business_recovered_value_est": float(estimated_recovered_value),
        "business_contact_cost_est": float(estimated_contact_cost),
        "business_net_value_est": float(estimated_net_value),
    }


def evaluate_model(
    model: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    config: BusinessMetricConfig,
) -> Dict[str, float]:
    y_score = model.predict_proba(x_test)[:, 1]
    y_pred = (y_score >= config.threshold).astype(int)

    metrics = {
        "auc_roc": float(roc_auc_score(y_test, y_score)),
        "pr_auc": float(average_precision_score(y_test, y_score)),
        "f1_score": float(f1_score(y_test, y_pred)),
    }
    metrics.update(compute_business_metric(y_test, y_score, config))
    return metrics


def build_model_pipelines(preprocessor: ColumnTransformer) -> Dict[str, Pipeline]:
    return {
        "dummy_classifier": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", DummyClassifier(strategy="most_frequent", random_state=42)),
            ]
        ),
        "logistic_regression": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    LogisticRegression(
                        max_iter=500,
                        solver="lbfgs",
                        n_jobs=None,
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
    metrics: Dict[str, float],
    x_test: pd.DataFrame,
    y_test: pd.Series,
    dataset_hash: str,
    dataset_path: Path,
    business_cfg: BusinessMetricConfig,
    output_dir: Path,
) -> None:
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
    # pipeline ponta a ponta: dados -> treino -> avaliação -> tracking
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
    model_pipelines = build_model_pipelines(preprocessor)
    business_cfg = BusinessMetricConfig()

    mlflow.set_experiment(experiment_name)

    rows = []
    for model_name, model_pipeline in model_pipelines.items():
        log_event(
            LOGGER,
            "model_training_started",
            event="model_training_started",
            model=model_name,
            step="train",
        )

        model_pipeline.fit(x_train, y_train)
        metrics = evaluate_model(model_pipeline, x_test, y_test, business_cfg)

        log_run_to_mlflow(
            model_name=model_name,
            model_pipeline=model_pipeline,
            metrics=metrics,
            x_test=x_test,
            y_test=y_test,
            dataset_hash=dataset_hash,
            dataset_path=Path("data/raw/telco_customer_churn.csv"),
            business_cfg=business_cfg,
            output_dir=Path("models/reports"),
        )

        row = {"model": model_name, "dataset_version_hash": dataset_hash, **metrics}
        rows.append(row)

        log_event(
            LOGGER,
            "model_training_completed",
            event="model_training_completed",
            model=model_name,
            step="evaluate",
        )

    metrics_df = pd.DataFrame(rows).sort_values(by="auc_roc", ascending=False)
    Path("models").mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv("models/baseline_metrics.csv", index=False)

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