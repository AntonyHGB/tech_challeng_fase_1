"""Schemas pandera para validação de DataFrames do dataset Telco Churn.

Pydantic valida payloads HTTP da API; pandera valida os DataFrames
tabulares usados em treino, batch e ingestão. As duas camadas são
complementares e garantem qualidade de dados em pontos diferentes
do pipeline.
"""

from __future__ import annotations

import pandera.pandas as pa
from pandera.pandas import Column, DataFrameSchema

CATEGORICAL_YES_NO = ["Yes", "No"]
INTERNET_SERVICE_OPTIONS = ["DSL", "Fiber optic", "No"]
MULTIPLE_LINES_OPTIONS = ["Yes", "No", "No phone service"]
ADDON_OPTIONS = ["Yes", "No", "No internet service"]
CONTRACT_OPTIONS = ["Month-to-month", "One year", "Two year"]
PAYMENT_METHOD_OPTIONS = [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]
GENDER_OPTIONS = ["Male", "Female"]


TELCO_RAW_SCHEMA = DataFrameSchema(
    columns={
        "gender": Column(str, pa.Check.isin(GENDER_OPTIONS), nullable=False),
        "SeniorCitizen": Column(int, pa.Check.isin([0, 1]), nullable=False),
        "Partner": Column(str, pa.Check.isin(CATEGORICAL_YES_NO), nullable=False),
        "Dependents": Column(str, pa.Check.isin(CATEGORICAL_YES_NO), nullable=False),
        "tenure": Column(int, pa.Check.ge(0), nullable=False),
        "PhoneService": Column(str, pa.Check.isin(CATEGORICAL_YES_NO), nullable=False),
        "MultipleLines": Column(str, pa.Check.isin(MULTIPLE_LINES_OPTIONS), nullable=False),
        "InternetService": Column(str, pa.Check.isin(INTERNET_SERVICE_OPTIONS), nullable=False),
        "OnlineSecurity": Column(str, pa.Check.isin(ADDON_OPTIONS), nullable=False),
        "OnlineBackup": Column(str, pa.Check.isin(ADDON_OPTIONS), nullable=False),
        "DeviceProtection": Column(str, pa.Check.isin(ADDON_OPTIONS), nullable=False),
        "TechSupport": Column(str, pa.Check.isin(ADDON_OPTIONS), nullable=False),
        "StreamingTV": Column(str, pa.Check.isin(ADDON_OPTIONS), nullable=False),
        "StreamingMovies": Column(str, pa.Check.isin(ADDON_OPTIONS), nullable=False),
        "Contract": Column(str, pa.Check.isin(CONTRACT_OPTIONS), nullable=False),
        "PaperlessBilling": Column(str, pa.Check.isin(CATEGORICAL_YES_NO), nullable=False),
        "PaymentMethod": Column(str, pa.Check.isin(PAYMENT_METHOD_OPTIONS), nullable=False),
        "MonthlyCharges": Column(float, pa.Check.ge(0), nullable=False),
        "TotalCharges": Column(float, pa.Check.ge(0), nullable=True),
        "Churn": Column(str, pa.Check.isin(CATEGORICAL_YES_NO), nullable=False),
    },
    strict=False,
    coerce=True,
)


TELCO_FEATURES_SCHEMA = DataFrameSchema(
    columns={
        col: spec for col, spec in TELCO_RAW_SCHEMA.columns.items() if col != "Churn"
    },
    strict=False,
    coerce=True,
)
