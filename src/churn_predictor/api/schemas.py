"""Schemas Pydantic para validação de entrada e saída da API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Payload de entrada para o endpoint /predict."""

    gender: str = Field(..., examples=["Female"])
    senior_citizen: int = Field(..., ge=0, le=1, alias="SeniorCitizen", examples=[0])
    partner: str = Field(..., alias="Partner", examples=["Yes"])
    dependents: str = Field(..., alias="Dependents", examples=["No"])
    tenure: int = Field(..., ge=0, examples=[1])
    phone_service: str = Field(..., alias="PhoneService", examples=["No"])
    multiple_lines: str = Field(..., alias="MultipleLines", examples=["No phone service"])
    internet_service: str = Field(..., alias="InternetService", examples=["DSL"])
    online_security: str = Field(..., alias="OnlineSecurity", examples=["No"])
    online_backup: str = Field(..., alias="OnlineBackup", examples=["Yes"])
    device_protection: str = Field(..., alias="DeviceProtection", examples=["No"])
    tech_support: str = Field(..., alias="TechSupport", examples=["No"])
    streaming_tv: str = Field(..., alias="StreamingTV", examples=["No"])
    streaming_movies: str = Field(..., alias="StreamingMovies", examples=["No"])
    contract: str = Field(..., alias="Contract", examples=["Month-to-month"])
    paperless_billing: str = Field(..., alias="PaperlessBilling", examples=["Yes"])
    payment_method: str = Field(..., alias="PaymentMethod", examples=["Electronic check"])
    monthly_charges: float = Field(..., alias="MonthlyCharges", ge=0, examples=[29.85])
    total_charges: float = Field(..., alias="TotalCharges", ge=0, examples=[29.85])

    model_config = {"populate_by_name": True}


class PredictionResponse(BaseModel):
    """Payload de saída do endpoint /predict."""

    churn_probability: float = Field(..., ge=0, le=1, description="Probabilidade de churn.")
    churn_prediction: int = Field(..., ge=0, le=1, description="1 = churn, 0 = não churn.")
    model_used: str = Field(..., description="Nome do modelo utilizado na inferência.")
