from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field
from starlette.responses import JSONResponse

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("churn_api")
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv(
    "MODEL_PATH",
    str(BASE_DIR.parent.parent / "artifacts" / "churnguard_pipeline.joblib"),
))

# Feature order taken from the supplied notebook's training code.
# Status and Churn are not model inputs. Avg_Usage_Duration is engineered below.
MODEL_FEATURES = [
    "Call  Failure",
    "Complains",
    "Subscription  Length",
    "Charge  Amount",
    "Seconds of Use",
    "Frequency of use",
    "Frequency of SMS",
    "Distinct Called Numbers",
    "Age Group",
    "Tariff Plan",
    "Age",
    "Customer Value",
    "Avg_Usage_Duration",
]

class CustomerInput(BaseModel):
    """Customer fields accepted by the API; engineered features are server-side."""
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    call_failure: int = Field(alias="Call  Failure", ge=0, le=36, description="Observed dataset range: 0–36")
    complains: int = Field(alias="Complains", ge=0, le=1, description="0 = no complaint, 1 = complaint")
    subscription_length: int = Field(alias="Subscription  Length", ge=3, le=47, description="Observed dataset range: 3–47")
    charge_amount: int = Field(alias="Charge  Amount", ge=0, le=10, description="Values observed in training data: 0–10")
    seconds_of_use: int = Field(alias="Seconds of Use", ge=0, le=17090, description="Observed dataset range: 0–17090")
    frequency_of_use: int = Field(alias="Frequency of use", ge=0, le=255, description="Observed dataset range: 0–255")
    frequency_of_sms: int = Field(alias="Frequency of SMS", ge=0, le=522, description="Observed dataset range: 0–522")
    distinct_called_numbers: int = Field(alias="Distinct Called Numbers", ge=0, le=97, description="Observed dataset range: 0–97")
    age_group: int = Field(alias="Age Group", ge=1, le=5, description="Supported groups observed in training data: 1–5")
    tariff_plan: int = Field(alias="Tariff Plan", ge=1, le=2, description="Supported plans observed in training data: 1 or 2")
    age: int = Field(alias="Age", ge=15, le=55, description="Observed dataset range: 15–55")
    customer_value: float = Field(alias="Customer Value", ge=0, le=2165.28, description="Observed dataset range: 0–2165.28")

class PredictionResponse(BaseModel):
    predicted_class: int
    churn_label: str
    churn_probability: float | None = None
    probability_note: str | None = None

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool

def load_model(path: Path) -> Any:
    """Load a trusted, locally supplied joblib artifact. Never load artifacts from users."""
    if not path.is_file():
        raise FileNotFoundError(f"Model artifact is missing: {path.name}")
    # joblib/pickle can execute code while loading. Only load your own trusted artifact.
    model = joblib.load(path)
    if not callable(getattr(model, "predict", None)):
        raise TypeError("The loaded artifact does not expose predict().")
    return model

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.model = load_model(MODEL_PATH)
        logger.info("Model artifact loaded successfully.")
    except Exception as exc:
        app.state.model = None
        logger.exception("Could not load model artifact (%s).", type(exc).__name__)
    yield

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predicts customer churn using the project's saved scikit-learn model.",
    version="1.0.0",
    lifespan=lifespan,
)

@app.exception_handler(Exception)
async def unexpected_error_handler(request: Request, exc: Exception):
    logger.exception("Unexpected API error at %s (%s).", request.url.path, type(exc).__name__)
    return JSONResponse(status_code=500, content={"detail": "An unexpected server error occurred."})

@app.get("/", tags=["General"])
def root():
    """Basic API information."""
    return {
        "message": "Customer Churn Prediction API",
        "docs": "/docs",
        "health": "/health",
        "prediction_endpoint": "POST /predict",
    }

@app.get("/health", response_model=HealthResponse, tags=["General"])
def health(request: Request):
    """Readiness check: returns 503 if the model could not be loaded."""
    model_loaded = getattr(request.app.state, "model", None) is not None
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model is unavailable; service is not ready.")
    return HealthResponse(status="ready", model_loaded=True)

@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Predict customer churn",
)
def predict(payload: CustomerInput, request: Request):
    """Validate customer data, create the engineered feature, and run inference."""
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Model is unavailable; prediction cannot be made.")

    values = payload.model_dump(by_alias=True)
    values["Avg_Usage_Duration"] = (
        values["Seconds of Use"] / (values["Frequency of use"] + 1)
    )

    # Use the known training feature order; if the estimator exposes feature_names_in_,
    # verify that its expected schema agrees before passing data to it.
    expected = list(getattr(model, "feature_names_in_", MODEL_FEATURES))
    if set(expected) != set(MODEL_FEATURES) or len(expected) != len(MODEL_FEATURES):
        logger.error("Saved model feature schema differs from the notebook schema.")
        raise HTTPException(status_code=500, detail="The saved model schema is incompatible with this API.")

    frame = pd.DataFrame([{name: values[name] for name in expected}], columns=expected)

    try:
        predicted = int(model.predict(frame)[0])
        churn_probability = None
        probability_note = None

        if callable(getattr(model, "predict_proba", None)):
            classes = list(getattr(model, "classes_", []))
            if not classes and hasattr(model, "named_steps"):
                final_estimator = list(model.named_steps.values())[-1]
                classes = list(getattr(final_estimator, "classes_", []))

            if 1 in classes:
                probabilities = model.predict_proba(frame)[0]
                churn_probability = float(probabilities[classes.index(1)])
                probability_note = "Estimated model probability, not a guaranteed outcome."

        return PredictionResponse(
            predicted_class=predicted,
            churn_label="Churn" if predicted == 1 else "No Churn",
            churn_probability=churn_probability,
            probability_note=probability_note,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Model inference failed (%s).", type(exc).__name__)
        raise HTTPException(status_code=500, detail="Prediction failed due to an internal inference error.") from exc
