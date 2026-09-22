"""FastAPI serving layer for the sentiment models.

Start it with::

    uvicorn api.main:app --host 0.0.0.0 --port 8000

The service distinguishes two kinds of failure, because they mean different
things to whatever is in front of it:

* **A model is not loadable here** - its weights are missing or are Git LFS
  pointer stubs. That is a 503: the request was fine, this instance just cannot
  serve it, and a different instance might.
* **Anything else** is a genuine 500.

Models load lazily on first use rather than at startup, so the container becomes
healthy quickly and a broken artifact cannot prevent the process from starting.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, field_validator
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from api.observability import configure_logging, request_id_middleware
from api.rate_limit import build_limiter, rate_limit_exceeded_handler
from src.config import MODEL_ALIASES, MODEL_KEYS, resolve_model
from src.inference.predictor import (
    ModelUnavailableError,
    Prediction,
    available_models,
    load_predictor,
    model_status,
)

LOGGER = logging.getLogger(__name__)

MAX_BATCH_SIZE = 256

configure_logging()

app = FastAPI(
    title="Sentiment Analysis API",
    description=(
        "Three-class tweet sentiment classification: negative, neutral, positive."
    ),
    version="1.0.0",
)

# Correlates every log line for a request and echoes X-Request-ID back, so a
# caller can quote an id when reporting a problem.
app.middleware("http")(request_id_middleware)

# Per-IP rate limiting. A backstop for a single process, not a substitute for
# limiting at the edge - see api/rate_limit.py.
limiter = build_limiter()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)


class PredictRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5_000)
    model: str = Field(
        default="lr",
        description=(
            f"One of: {', '.join(MODEL_KEYS)}. "
            f"Deprecated aliases: {', '.join(MODEL_ALIASES)}."
        ),
    )

    @field_validator("model")
    @classmethod
    def known_model(cls, value: str) -> str:
        # Deprecated aliases resolve to the canonical key, so a request written
        # against the old name still works.
        resolved = resolve_model(value)
        if resolved not in MODEL_KEYS:
            raise ValueError(f"must be one of: {', '.join(MODEL_KEYS)}")
        return resolved


class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=MAX_BATCH_SIZE)
    model: str = Field(default="lr")

    @field_validator("model")
    @classmethod
    def known_model(cls, value: str) -> str:
        # Deprecated aliases resolve to the canonical key, so a request written
        # against the old name still works.
        resolved = resolve_model(value)
        if resolved not in MODEL_KEYS:
            raise ValueError(f"must be one of: {', '.join(MODEL_KEYS)}")
        return resolved


class PredictResponse(BaseModel):
    model: str
    # Content-derived version of the artifact that produced this prediction.
    version: str | None = None
    label: str
    confidence: float
    probabilities: dict[str, float]

    @classmethod
    def from_prediction(cls, prediction: Prediction) -> PredictResponse:
        return cls(
            model=prediction.model,
            version=prediction.version,
            label=prediction.label,
            confidence=prediction.confidence,
            probabilities=prediction.probabilities,
        )


class BatchPredictResponse(BaseModel):
    predictions: list[PredictResponse]


def _load_or_503(model: str):
    """Load a predictor, translating unavailability into 503."""
    try:
        return load_predictor(model)
    except ModelUnavailableError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)
        ) from error


@app.get("/health")
@limiter.exempt
def health() -> dict:
    """Liveness: the process is up. Says nothing about the models."""
    return {"status": "ok", "models_available": available_models()}


@app.get("/ready")
@limiter.exempt
def ready() -> dict:
    """Readiness: at least one model can actually be served."""
    models = available_models()
    if not models:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "No usable model artifacts. Run `git lfs install && git lfs pull`, "
                "or train a model with `python -m src.training.train_lr`."
            ),
        )
    return {"status": "ready", "models_available": models}


@app.get("/models")
@limiter.exempt
def models() -> dict:
    """Every known model and why it is or is not available."""
    return {"models": model_status()}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """Classify a single text."""
    predictor = _load_or_503(request.model)
    prediction = predictor.predict(request.text)

    # The text itself is never logged - see api/observability.py.
    LOGGER.info(
        "prediction",
        extra={
            "model": prediction.model,
            "model_version": prediction.version,
            "label": prediction.label,
            "confidence": round(prediction.confidence, 4),
            "text_length": len(request.text),
        },
    )
    return PredictResponse.from_prediction(prediction)


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(request: BatchPredictRequest) -> BatchPredictResponse:
    """Classify up to MAX_BATCH_SIZE texts in one call."""
    predictor = _load_or_503(request.model)
    predictions = [predictor.predict(text) for text in request.texts]

    LOGGER.info(
        "batch prediction",
        extra={
            "model": request.model,
            "model_version": predictor.version,
            "batch_size": len(predictions),
        },
    )
    return BatchPredictResponse(
        predictions=[PredictResponse.from_prediction(p) for p in predictions]
    )
