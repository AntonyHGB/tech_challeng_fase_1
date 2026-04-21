"""Testes para o módulo data.py."""

from __future__ import annotations

import pandas as pd
import pytest

from churn_predictor.data import clean_telco_dataset, dataset_version_hash, split_xy


def test_clean_telco_dataset_removes_customer_id():
    """Garante que a coluna customerID é removida se existir."""
    df = pd.DataFrame(
        {"customerID": ["123", "456"], "TotalCharges": ["10", "20"], "Churn": ["Yes", "No"]}
    )
    cleaned_df = clean_telco_dataset(df)
    assert "customerID" not in cleaned_df.columns


def test_clean_telco_dataset_converts_total_charges():
    """Garante que TotalCharges vira numérico e espaços viram NaN."""
    df = pd.DataFrame({"TotalCharges": ["10.5", " ", "20.0"], "Churn": ["Yes", "No", "Yes"]})
    cleaned_df = clean_telco_dataset(df)
    assert pd.api.types.is_numeric_dtype(cleaned_df["TotalCharges"])
    assert cleaned_df["TotalCharges"].isna().sum() == 1


def test_clean_telco_dataset_raises_on_missing_required_columns():
    """Garante que levanta ValueError se faltar Churn ou TotalCharges."""
    df = pd.DataFrame({"customerID": ["123"], "gender": ["Female"]})
    with pytest.raises(ValueError, match="O dataset não possui as colunas obrigatórias"):
        clean_telco_dataset(df)


def test_split_xy_valid():
    """Garante que o split funciona com mapeamento Yes/No correto e insensível a case."""
    df = pd.DataFrame({"feature1": [1, 2, 3], "Churn": ["Yes", "no", " YES "]})
    x, y = split_xy(df)

    assert "Churn" not in x.columns
    assert list(y) == [1, 0, 1]


def test_split_xy_invalid_target():
    """Garante erro ao encontrar classes desconhecidas no target."""
    df = pd.DataFrame({"feature1": [1], "Churn": ["Maybe"]})
    with pytest.raises(ValueError, match="Valores inesperados encontrados"):
        split_xy(df)


def test_dataset_version_hash():
    """Garante que o hash é determinístico para o mesmo dataframe."""
    df1 = pd.DataFrame({"col1": [1, 2], "col2": ["A", "B"]})
    df2 = pd.DataFrame({"col1": [1, 2], "col2": ["A", "B"]})
    df3 = pd.DataFrame({"col1": [2, 1], "col2": ["B", "A"]})

    assert dataset_version_hash(df1) == dataset_version_hash(df2)
    assert dataset_version_hash(df1) != dataset_version_hash(df3)
