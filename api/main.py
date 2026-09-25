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
``PRELOAD_MODELS`` opts into warming some or all of them up before the first
request; a model that fails to load then is logged and skipped, so the same
guarantee holds.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, HTTPException, Request, status
from pydantic import AfterValidator, BaseModel, Field, field_validator
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from api.observability import configure_logging, request_id_middleware
from api.rate_limit import (
    build_limiter,
    build_text_budget,
    client_key,
    rate_limit_exceeded_handler,
)
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


def preload_list() -> list[str]:
    """Model keys named by ``PRELOAD_MODELS``: comma-separated, or ``all``."""
    raw = os.environ.get("PRELOAD_MODELS", "").strip()
    if not raw:
        return []
    if raw.lower() == "all":
        return list(MODEL_KEYS)
    return [name.strip() for name in raw.split(",") if name.strip()]


def preload_models(names: list[str]) -> list[str]:
    """Load models ahead of the first request; returns the keys that loaded.

    Never fatal. Startup must not depend on an artifact being present or
    loadable - that is the guarantee lazy loading gives - so an unknown name,
    a missing artifact or a loader that raises is logged and skipped.
    """
    loaded = []
    for name in names:
        try:
            predictor = load_predictor(name)
        except (ModelUnavailableError, KeyError) as error:
            LOGGER.warning("PRELOAD_MODELS: skipped %s: %s", name, str(error).strip("'"))
        except Exception:
            LOGGER.exception("PRELOAD_MODELS: %s failed to load and was skipped", name)
        else:
            loaded.append(predictor.key)
            LOGGER.info(
                "preloaded model",
                extra={"model": predictor.key, "model_version": predictor.version},
            )
    return loaded


@asynccontextmanager
async def lifespan(app: FastAPI):
    preload_models(preload_list())
    yield


app = FastAPI(
    title="Sentiment Analysis API",
    description=(
        "Three-class tweet sentiment classification: negative, neutral, positive."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Per-IP rate limiting. A backstop for a single process, not a substitute for
# limiting at the edge - see api/rate_limit.py.
limiter = build_limiter()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# The request limiter charges one unit per call, so a batch of 256 texts costs
# the same as one text. This second budget is counted in texts and charged by
# the prediction endpoints themselves. Kept on app.state, like the limiter, so
# tests can swap in a differently sized budget without rebuilding the app.
app.state.text_budget = build_text_budget()

# FastAPI owns the docs routes, so they cannot carry @limiter.exempt; name
# them directly. Reading the documentation should not consume a caller's budget.
limiter._exempt_routes.update(
    {
        "fastapi.applications.openapi",
        "fastapi.applications.swagger_ui_html",
        "fastapi.applications.swagger_ui_redirect",
        "fastapi.applications.redoc_html",
    }
)

# Registered last, so it is the OUTERMOST middleware: Starlette runs the most
# recently added first. A throttled request must still get an X-Request-ID and
# an access log line - otherwise the requests most worth investigating are the
# ones that leave no trace.
app.middleware("http")(request_id_middleware)


# Shared so single and batch requests cannot drift apart in what they accept.
TEXT_FIELD = Field(min_length=1, max_length=5_000)


def _non_blank(value: str) -> str:
    """Reject whitespace-only input.

    `min_length` counts characters, so "   " passed validation and was scored -
    the model duly returned a confident-looking label for nothing.
    """
    if not value.strip():
        raise ValueError("must not be blank")
    return value


NonBlankText = Annotated[str, TEXT_FIELD, AfterValidator(_non_blank)]


class PredictRequest(BaseModel):
    text: NonBlankText
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
    # Annotated applies the length limits to each item; putting them on the
    # list alone would bound how many texts arrive, not how long each is.
    texts: list[NonBlankText] = Field(min_length=1, max_length=MAX_BATCH_SIZE)
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


def _charge_texts(request: Request, cost: int) -> None:
    """Spend ``cost`` texts of the caller's budget, or answer 429.

    Charged before the model is loaded or run, so a throttled caller cannot
    make the process do the expensive part anyway. The response mirrors the
    request limiter's: a ``detail`` message and a ``Retry-After`` header.
    """
    budget = request.app.state.text_budget
    if not budget.enabled:
        return

    if not budget.fits(cost):
        # Waiting would not help: the whole window is smaller than this batch.
        LOGGER.warning(
            "batch exceeds text budget",
            extra={"path": request.url.path, "limit": budget.limit, "texts": cost},
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                f"A batch of {cost} texts can never fit under the text rate limit "
                f"of {budget.limit}; send at most {budget.capacity} texts per batch."
            ),
        )

    retry_after = budget.consume(client_key(request), cost)
    if retry_after is not None:
        LOGGER.warning(
            "text rate limit exceeded",
            extra={"path": request.url.path, "limit": budget.limit, "texts": cost},
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Text rate limit exceeded: {budget.limit}.",
            headers={"Retry-After": str(retry_after)},
        )


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
                "No usable model artifacts. Run `make fetch-weights` to download "
                "them, or train a model with `python -m src.training.train_lr`."
            ),
        )
    return {"status": "ready", "models_available": models}


@app.get("/models")
@limiter.exempt
def models() -> dict:
    """Every known model and why it is or is not available."""
    return {"models": model_status()}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest, request: Request) -> PredictResponse:
    """Classify a single text."""
    _charge_texts(request, 1)
    predictor = _load_or_503(payload.model)
    prediction = predictor.predict(payload.text)

    # The text itself is never logged - see api/observability.py.
    LOGGER.info(
        "prediction",
        extra={
            "model": prediction.model,
            "model_version": prediction.version,
            "label": prediction.label,
            "confidence": round(prediction.confidence, 4),
            "text_length": len(payload.text),
        },
    )
    return PredictResponse.from_prediction(prediction)


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(payload: BatchPredictRequest, request: Request) -> BatchPredictResponse:
    """Classify up to MAX_BATCH_SIZE texts in one call."""
    _charge_texts(request, len(payload.texts))
    predictor = _load_or_503(payload.model)
    predictions = predictor.predict_many(payload.texts)

    LOGGER.info(
        "batch prediction",
        extra={
            "model": payload.model,
            "model_version": predictor.version,
            "batch_size": len(predictions),
        },
    )
    return BatchPredictResponse(
        predictions=[PredictResponse.from_prediction(p) for p in predictions]
    )
