"""Structured logging and request correlation for the API.

Logs are emitted as one JSON object per line, so they can be shipped and queried
without anyone writing a regex for a message format that will change. Every line
carries the ``request_id`` of the request being served, which makes a single
prediction traceable end to end: the access line, any warning raised while
serving it, and the prediction itself all share one identifier.

Paired with the model ``version`` on each prediction, a support question of the
form "why did it say that?" is answerable from logs alone - which request, which
artifact, which label, how confident.

**What is deliberately not logged: the text being classified.** It is user
content, it is frequently personal, and it is exactly the sort of thing that
ends up in a log aggregator for years. The length is recorded instead, which is
enough to investigate truncation and payload-size problems.

Configure with ``LOG_LEVEL`` (default INFO) and ``LOG_FORMAT`` (``json``, the
default, or ``plain`` for human-readable local runs).
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from contextvars import ContextVar

from starlette.requests import Request

REQUEST_ID_HEADER = "X-Request-ID"

# An inbound id is attacker-controlled and gets written into a JSON log line, so
# it is constrained rather than trusted: a newline in it would forge log entries.
SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)

# Attributes the stdlib puts on every record; anything else was passed by the
# caller as `extra` and belongs in the structured output.
_STANDARD_FIELDS = frozenset(logging.LogRecord("", 0, "", 0, "", None, None).__dict__) | {
    "message",
    "asctime",
    "taskName",
}


def get_request_id() -> str | None:
    """Request id for the request currently being served, if any."""
    return _request_id.get()


def set_request_id(value: str | None) -> None:
    _request_id.set(value)


def new_request_id() -> str:
    return uuid.uuid4().hex


def normalise_request_id(value: str | None) -> str:
    """Accept a caller-supplied id, or mint one if it is absent or unsafe."""
    if value and SAFE_REQUEST_ID.match(value):
        return value
    return new_request_id()


class RequestIdFilter(logging.Filter):
    """Attach the current request id to every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id()
        return True


class JsonFormatter(logging.Formatter):
    """One JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", None),
        }

        for key, value in record.__dict__.items():
            if key not in _STANDARD_FIELDS and key != "request_id":
                payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


def configure_logging(level: str | None = None, fmt: str | None = None) -> None:
    """Install the formatter and request-id filter on the root logger."""
    level = (level or os.environ.get("LOG_LEVEL", "INFO")).upper()
    fmt = (fmt or os.environ.get("LOG_FORMAT", "json")).lower()

    handler = logging.StreamHandler()
    handler.addFilter(RequestIdFilter())
    handler.setFormatter(
        JsonFormatter()
        if fmt == "json"
        else logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s [%(request_id)s] %(message)s"
        )
    )

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)


async def request_id_middleware(request: Request, call_next):
    """Correlate every log line for a request, and log the request itself."""
    logger = logging.getLogger("api.access")

    request_id = normalise_request_id(request.headers.get(REQUEST_ID_HEADER))
    token = _request_id.set(request_id)
    started = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        logger.exception(
            "request failed",
            extra={
                "method": request.method,
                "path": request.url.path,
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
            },
        )
        _request_id.reset(token)
        raise

    duration_ms = round((time.perf_counter() - started) * 1000, 2)
    response.headers[REQUEST_ID_HEADER] = request_id

    logger.info(
        "request",
        extra={
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "duration_ms": duration_ms,
        },
    )
    _request_id.reset(token)
    return response
