"""FastAPI service for the educational heart disease classifier."""

from __future__ import annotations

import json
import logging
import time
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import Response

from src.data import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    PROJECT_ROOT,
)

# ---------------------------------------------------------------------
# Configuration and model loading
# ---------------------------------------------------------------------

MODEL_PATH = PROJECT_ROOT / "artifacts" / "models" / "heart_disease_best_model.joblib"
METRICS_PATH = PROJECT_ROOT / "reports" / "model_metrics.json"
FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("heart_disease_api")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Trained model not found at {MODEL_PATH}. Run 'python -m src.train' first."
    )

model = joblib.load(MODEL_PATH)

if METRICS_PATH.exists():
    with METRICS_PATH.open(encoding="utf-8") as file:
        training_metadata = json.load(file)
    MODEL_NAME = training_metadata.get("selected_model", "Heart Disease Classifier")
else:
    MODEL_NAME = "Heart Disease Classifier"

app = FastAPI(
    title="Heart Disease Prediction API",
    description=(
        "Educational demonstration of a trained ML classification "
        "pipeline. This is not a medical diagnostic tool."
    ),
    version="1.0.0",
)


# ---------------------------------------------------------------------
# Request and response schemas
# ---------------------------------------------------------------------


class HeartDiseaseInput(BaseModel):
    """Input fields corresponding to the UCI Cleveland features."""

    model_config = ConfigDict(extra="forbid")

    age: int = Field(..., ge=1, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="Dataset code: 0 or 1")
    cp: int = Field(..., ge=1, le=4, description="Chest-pain category")
    trestbps: float = Field(..., gt=0, le=300, description="Resting blood pressure")
    chol: float = Field(..., gt=0, le=700, description="Serum cholesterol")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar code")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG category")
    thalach: float = Field(..., gt=0, le=250, description="Maximum heart rate")
    exang: int = Field(..., ge=0, le=1, description="Exercise-induced angina code")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression")
    slope: int = Field(..., ge=1, le=3, description="ST segment slope category")
    ca: int = Field(..., ge=0, le=3, description="Major vessels category")
    thal: Literal[3, 6, 7] = Field(..., description="Thal category")


class PredictionResponse(BaseModel):
    """Structured prediction response."""

    prediction: int
    label: str
    probability_heart_disease: float
    confidence: float
    model_name: str
    disclaimer: str


# ---------------------------------------------------------------------
# Request logging: record request metadata, not patient feature values
# ---------------------------------------------------------------------


@app.middleware("http")
async def log_requests(
    request: Request,
    call_next: RequestResponseEndpoint,
) -> Response:
    """Log request method, endpoint, status, and duration."""
    start = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.exception(
            "request_failed method=%s path=%s duration_ms=%.2f",
            request.method,
            request.url.path,
            elapsed_ms,
        )
        raise

    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "request_completed method=%s path=%s status_code=%s duration_ms=%.2f",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


# ---------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------


@app.get("/")
def root() -> dict[str, str]:
    """Describe the service and point users to interactive docs."""
    return {
        "service": "Heart Disease Prediction API",
        "status": "ready",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health() -> dict[str, str | bool]:
    """Report whether the model loaded into this process."""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_name": MODEL_NAME,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: HeartDiseaseInput) -> PredictionResponse:
    """Return the model's binary prediction and probability estimates."""
    features = pd.DataFrame([payload.model_dump()])
    features = features[FEATURE_COLUMNS]

    prediction = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]

    classes = list(model.classes_)
    positive_index = classes.index(1)
    predicted_index = classes.index(prediction)

    probability_positive = float(probabilities[positive_index])
    confidence = float(probabilities[predicted_index])

    logger.info(
        "prediction_completed model=%s prediction=%s",
        MODEL_NAME,
        prediction,
    )

    return PredictionResponse(
        prediction=prediction,
        label="positive" if prediction == 1 else "negative",
        probability_heart_disease=probability_positive,
        confidence=confidence,
        model_name=MODEL_NAME,
        disclaimer=(
            "Educational demonstration only; not intended for medical "
            "diagnosis or treatment decisions."
        ),
    )
