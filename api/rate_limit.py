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

Requests are the wrong unit for the prediction endpoints, though: a batch of
256 texts costs the same request as a single text, so the request limit alone
bounds calls, not work. A second budget, :class:`TextBudget`, is therefore
counted in texts. ``/predict`` costs 1 and ``/predict/batch`` costs one per
text, and a batch that would overrun the budget is rejected whole without
consuming anything.

Configure with ``RATE_LIMIT`` (e.g. ``60/minute``, or ``off`` to disable),
``TEXT_RATE_LIMIT`` (e.g. ``1024/minute``, or ``off``),
``RATE_LIMIT_STORAGE_URI`` and ``TRUST_PROXY_HEADERS``.
"""

from __future__ import annotations

import logging
import math
import os
import time

from limits import parse
from limits.storage import storage_from_string
from limits.strategies import MovingWindowRateLimiter
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.requests import Request
from starlette.responses import JSONResponse

LOGGER = logging.getLogger(__name__)

DEFAULT_LIMIT = "60/minute"
DEFAULT_TEXT_LIMIT = "1024/minute"

OFF_VALUES = {"off", "none", "disabled", ""}


def _limit_from_env(name: str, default: str) -> str | None:
    value = os.environ.get(name, default).strip()
    if value.lower() in OFF_VALUES:
        return None
    return value


def rate_limit() -> str | None:
    """Configured request limit, or None when rate limiting is switched off."""
    return _limit_from_env("RATE_LIMIT", DEFAULT_LIMIT)


def text_rate_limit() -> str | None:
    """Configured per-text limit, or None when it is switched off."""
    return _limit_from_env("TEXT_RATE_LIMIT", DEFAULT_TEXT_LIMIT)


def storage_uri() -> str:
    return os.environ.get("RATE_LIMIT_STORAGE_URI", "memory://")


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
    storage = storage_uri()

    if limit is None:
        LOGGER.info("Rate limiting disabled (RATE_LIMIT=off).")
    else:
        LOGGER.info(
            "Rate limiting enabled",
            extra={"limit": limit, "storage": storage.split("://")[0]},
        )

    return Limiter(
        key_func=client_key,
        default_limits=[limit] if limit else [],
        # A shared budget across every endpoint. Per-route limits meant one
        # caller got the full allowance on /predict *and* again on
        # /predict/batch. This counts requests only: a batch still costs one
        # request however many texts it carries, which is what TextBudget
        # below is for.
        application_limits=[limit] if limit else [],
        storage_uri=storage,
        enabled=limit is not None,
        # Tell callers where they stand before they hit the wall.
        headers_enabled=True,
    )


class TextBudget:
    """Budget counted in texts, so a batch of N costs N.

    Built directly on ``limits`` (the package slowapi itself uses) rather than
    on slowapi, whose decorators only know how to charge one unit per request.
    A moving window is used so a caller cannot double their allowance by
    straddling a fixed-window boundary.
    """

    # Namespaces the storage keys, so the text counters never collide with the
    # request counters slowapi keeps in the same store under the same caller.
    NAMESPACE = "texts"

    def __init__(self, limit: str | None, storage_uri: str) -> None:
        self.limit = limit
        self.enabled = limit is not None
        if limit is None:
            LOGGER.info("Per-text rate limiting disabled (TEXT_RATE_LIMIT=off).")
            self.capacity = None
            return

        self._item = parse(limit)
        self._limiter = MovingWindowRateLimiter(storage_from_string(storage_uri))
        # The most texts any single request can ever be granted.
        self.capacity = self._item.amount
        LOGGER.info(
            "Per-text rate limiting enabled",
            extra={"limit": limit, "storage": storage_uri.split("://")[0]},
        )

    def fits(self, cost: int) -> bool:
        """Whether a request of this size can ever be admitted."""
        return self.capacity is None or cost <= self.capacity

    def consume(self, key: str, cost: int) -> int | None:
        """Consume ``cost`` units; return None if allowed, else Retry-After seconds.

        Tested before it is hit, so a rejected batch consumes nothing: otherwise
        one oversized batch could lock a caller out of single predictions too.
        """
        if not self.enabled:
            return None
        if self._limiter.test(self._item, self.NAMESPACE, key, cost=cost):
            self._limiter.hit(self._item, self.NAMESPACE, key, cost=cost)
            return None
        stats = self._limiter.get_window_stats(self._item, self.NAMESPACE, key)
        return max(1, math.ceil(stats.reset_time - time.time()))


def build_text_budget() -> TextBudget:
    return TextBudget(text_rate_limit(), storage_uri())


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
