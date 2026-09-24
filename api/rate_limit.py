"""Rate limiting for the prediction endpoints.

**This is a backstop, not the real defence.** Rate limiting belongs at the edge -
an ingress, API gateway or reverse proxy - where it applies before a request
reaches any application process and is shared across every replica. What follows
protects a single process from a single noisy caller, which is worth having and
is not the same thing.

Two consequences of that, stated rather than discovered later:

* With the default in-memory store, each worker keeps its own counters. Four
  uvicorn workers mean a caller gets four times the configured limit. Point
  ``RATE_LIMIT_STORAGE_URI`` at Redis to share state across processes.
* The caller is identified by IP. Behind a proxy every request appears to come
  from the proxy, so the limit would apply to all traffic at once. Setting
  ``TRUST_PROXY_HEADERS=1`` switches to the first hop of ``X-Forwarded-For`` -
  do that **only** when a proxy you control overwrites that header, because
  otherwise a caller can set it themselves and sidestep the limit entirely.

Health and readiness probes are exempt, via ``@limiter.exempt`` in api/main.py,
as are the OpenAPI docs. Throttling a liveness probe gets a healthy container
restarted, which converts a traffic spike into an outage; throttling ``/docs``
just makes the service look broken to someone reading it.

The limit is shared across endpoints rather than applied per route, so a caller
cannot spend it twice by alternating between ``/predict`` and
``/predict/batch``.

Configure with ``RATE_LIMIT`` (e.g. ``60/minute``, or ``off`` to disable),
``RATE_LIMIT_STORAGE_URI`` and ``TRUST_PROXY_HEADERS``.
"""

from __future__ import annotations

import logging
import os

from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.requests import Request
from starlette.responses import JSONResponse

LOGGER = logging.getLogger(__name__)

DEFAULT_LIMIT = "60/minute"


def rate_limit() -> str | None:
    """Configured limit, or None when rate limiting is switched off."""
    value = os.environ.get("RATE_LIMIT", DEFAULT_LIMIT).strip()
    if value.lower() in {"off", "none", "disabled", ""}:
        return None
    return value


def trust_proxy_headers() -> bool:
    return os.environ.get("TRUST_PROXY_HEADERS", "").lower() in {"1", "true", "yes"}


def client_key(request: Request) -> str:
    """Identify the caller for limiting purposes.

    X-Forwarded-For is only consulted when explicitly trusted, because it is
    client-supplied otherwise and would make the limit trivially bypassable by
    sending a different value on each request.
    """
    if trust_proxy_headers():
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return get_remote_address(request)


def build_limiter() -> Limiter:
    limit = rate_limit()
    storage_uri = os.environ.get("RATE_LIMIT_STORAGE_URI", "memory://")

    if limit is None:
        LOGGER.info("Rate limiting disabled (RATE_LIMIT=off).")
    else:
        LOGGER.info(
            "Rate limiting enabled",
            extra={"limit": limit, "storage": storage_uri.split("://")[0]},
        )

    return Limiter(
        key_func=client_key,
        default_limits=[limit] if limit else [],
        # A shared budget across every endpoint. Per-route limits meant one
        # caller got the full allowance on /predict *and* again on
        # /predict/batch - and a batch carries up to 256 texts, so the real
        # ceiling was orders of magnitude above the configured one.
        application_limits=[limit] if limit else [],
        storage_uri=storage_uri,
        enabled=limit is not None,
        # Tell callers where they stand before they hit the wall.
        headers_enabled=True,
    )


def retry_after_seconds(exc: RateLimitExceeded) -> int:
    """How long the caller should wait, from the window of the limit they hit."""
    try:
        return int(exc.limit.limit.GRANULARITY.seconds)
    except AttributeError:
        return 60


def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    """429 with a Retry-After header, and a log line recording the breach."""
    LOGGER.warning(
        "rate limit exceeded",
        extra={"path": request.url.path, "limit": str(exc.limit.limit)},
    )
    response = JSONResponse(
        status_code=429,
        content={"detail": f"Rate limit exceeded: {exc.limit.limit}."},
    )
    response.headers["Retry-After"] = str(retry_after_seconds(exc))
    return response
