"""Aplicação principal FastAPI para servir o modelo de churn."""

from __future__ import annotations

import logging
import time

from fastapi import FastAPI, Request

from churn_predictor.api.routers.predict import router as predict_router
from churn_predictor.logging_utils import configure_logging

configure_logging()
LOGGER = logging.getLogger(__name__)

app = FastAPI(
    title="Churn Predictor API",
    description="API de inferência para previsão de churn em telecomunicações.",
    version="0.1.0",
)


@app.middleware("http")
async def latency_middleware(request: Request, call_next):
    """Middleware que mede e loga a latência de cada request."""
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000

    LOGGER.info(
        "request_completed",
        extra={
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "latency_ms": round(elapsed_ms, 2),
        },
    )
    return response


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Endpoint de health check."""
    return {"status": "ok"}


app.include_router(predict_router)
