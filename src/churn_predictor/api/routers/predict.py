"""Router para o endpoint de predição."""

from __future__ import annotations

import logging

import pandas as pd
from fastapi import APIRouter

from churn_predictor.api.model_loader import load_model
from churn_predictor.api.schemas import PredictionRequest, PredictionResponse

LOGGER = logging.getLogger(__name__)

router = APIRouter()

FEATURE_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]

MODEL_NAME = "logistic_regression"


@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> PredictionResponse:
    """Recebe dados de um cliente e retorna a predição de churn."""
    pipeline = load_model()

    row = {
        "gender": request.gender,
        "SeniorCitizen": request.senior_citizen,
        "Partner": request.partner,
        "Dependents": request.dependents,
        "tenure": request.tenure,
        "PhoneService": request.phone_service,
        "MultipleLines": request.multiple_lines,
        "InternetService": request.internet_service,
        "OnlineSecurity": request.online_security,
        "OnlineBackup": request.online_backup,
        "DeviceProtection": request.device_protection,
        "TechSupport": request.tech_support,
        "StreamingTV": request.streaming_tv,
        "StreamingMovies": request.streaming_movies,
        "Contract": request.contract,
        "PaperlessBilling": request.paperless_billing,
        "PaymentMethod": request.payment_method,
        "MonthlyCharges": request.monthly_charges,
        "TotalCharges": request.total_charges,
    }

    input_df = pd.DataFrame([row], columns=FEATURE_COLUMNS)

    probabilities = pipeline.predict_proba(input_df)
    churn_prob = float(probabilities[0][1])

    LOGGER.info(
        "prediction_completed",
        extra={"churn_probability": churn_prob},
    )

    return PredictionResponse(
        churn_probability=round(churn_prob, 4),
        churn_prediction=int(churn_prob >= 0.5),
        model_used=MODEL_NAME,
    )
