from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Tuple

import pandas as pd

LOGGER = logging.getLogger(__name__)

DEFAULT_DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/"
    "data/Telco-Customer-Churn.csv"
)
DEFAULT_DATA_PATH = Path("data/raw/telco_customer_churn.csv")


def configure_data_dirs() -> None:
    """Ensure expected data directories exist."""
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)


def download_dataset_if_needed(
    data_path: Path = DEFAULT_DATA_PATH,
    url: str = DEFAULT_DATA_URL,
    force: bool = False,
) -> Path:
    """Download IBM Telco Churn dataset if missing."""
    configure_data_dirs()

    if data_path.exists() and not force:
        LOGGER.info(
            "dataset_download_skipped",
            extra={"data_path": str(data_path), "reason": "already_exists"},
        )
        return data_path

    LOGGER.info(
        "dataset_download_started",
        extra={"data_path": str(data_path), "url": url, "force": force},
    )
    df = pd.read_csv(url)
    df.to_csv(data_path, index=False)
    LOGGER.info(
        "dataset_download_completed",
        extra={"data_path": str(data_path), "rows": len(df), "columns": len(df.columns)},
    )
    return data_path


def load_dataset(data_path: Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load dataset from disk."""
    if not data_path.exists():
        raise FileNotFoundError(
            f"dataset not found at '{data_path}'. run download_dataset_if_needed first."
        )
    df = pd.read_csv(data_path)
    LOGGER.info(
        "dataset_loaded",
        extra={"data_path": str(data_path), "rows": len(df), "columns": len(df.columns)},
    )
    return df


def clean_telco_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Apply basic cleanup for IBM Telco dataset."""
    data = df.copy()

    if "TotalCharges" in data.columns:
        data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")

    if "customerID" in data.columns:
        data = data.drop(columns=["customerID"])

    return data


def split_xy(df: pd.DataFrame, target_col: str = "Churn") -> Tuple[pd.DataFrame, pd.Series]:
    """Split features and target label."""
    if target_col not in df.columns:
        raise ValueError(f"target column '{target_col}' not found in dataset")

    y = df[target_col].map({"Yes": 1, "No": 0}).astype(int)
    x = df.drop(columns=[target_col])
    return x, y


def dataset_version_hash(df: pd.DataFrame) -> str:
    """Create deterministic dataset hash for experiment tracking."""
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(csv_bytes).hexdigest()