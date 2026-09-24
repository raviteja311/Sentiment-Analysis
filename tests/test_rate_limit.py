"""Rate limiting.

The suite disables rate limiting globally (see conftest), so this module builds
its own limited app rather than relying on the shared one. That keeps a few
dozen unrelated API tests from throttling each other, and keeps the limits under
test explicit instead of inherited from the environment.
"""

import pytest

from api import rate_limit

pytest.importorskip("slowapi")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from slowapi.errors import RateLimitExceeded  # noqa: E402
from slowapi.middleware import SlowAPIMiddleware  # noqa: E402

from api.observability import REQUEST_ID_HEADER, request_id_middleware  # noqa: E402


def build_app(limit: str, **env) -> TestClient:
    """A minimal app wired exactly like api/main.py, middleware order included."""
    app = FastAPI()
    app.state.limiter = rate_limit.build_limiter()
    app.add_exception_handler(RateLimitExceeded, rate_limit.rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)
    # Added last, so it is outermost - see the ordering test below.
    app.middleware("http")(request_id_middleware)

    @app.get("/limited")
    def limited():
        return {"ok": True}

    @app.get("/probe")
    @app.state.limiter.exempt
    def probe():
        return {"ok": True}

    return TestClient(app)


@pytest.fixture
def limited_client(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT", "3/minute")
    monkeypatch.setenv("RATE_LIMIT_STORAGE_URI", "memory://")
    return build_app("3/minute")


# --- configuration ---------------------------------------------------------


def test_limiting_is_on_by_default(monkeypatch):
    monkeypatch.delenv("RATE_LIMIT", raising=False)
    assert rate_limit.rate_limit() == rate_limit.DEFAULT_LIMIT


@pytest.mark.parametrize("value", ["off", "OFF", "none", "disabled", ""])
def test_limiting_can_be_switched_off(monkeypatch, value):
    monkeypatch.setenv("RATE_LIMIT", value)
    assert rate_limit.rate_limit() is None


def test_a_custom_limit_is_honoured(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT", "5/second")
    assert rate_limit.rate_limit() == "5/second"


def test_a_disabled_limiter_reports_itself_disabled(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT", "off")
    assert rate_limit.build_limiter().enabled is False


# --- identifying the caller ------------------------------------------------


class FakeRequest:
    def __init__(self, headers=None, host="1.2.3.4"):
        self.headers = headers or {}
        self.client = type("Client", (), {"host": host})()


def test_the_client_ip_identifies_the_caller(monkeypatch):
    monkeypatch.delenv("TRUST_PROXY_HEADERS", raising=False)
    assert rate_limit.client_key(FakeRequest(host="9.9.9.9")) == "9.9.9.9"


def test_forwarded_headers_are_ignored_unless_trusted(monkeypatch):
    # Otherwise a caller sends a different X-Forwarded-For per request and the
    # limit means nothing.
    monkeypatch.delenv("TRUST_PROXY_HEADERS", raising=False)
    request = FakeRequest(headers={"X-Forwarded-For": "5.5.5.5"}, host="9.9.9.9")
    assert rate_limit.client_key(request) == "9.9.9.9"


def test_forwarded_headers_are_used_when_trusted(monkeypatch):
    monkeypatch.setenv("TRUST_PROXY_HEADERS", "1")
    request = FakeRequest(headers={"X-Forwarded-For": "5.5.5.5, 10.0.0.1"})
    assert rate_limit.client_key(request) == "5.5.5.5"


def test_trusted_mode_falls_back_to_the_socket_when_the_header_is_absent(monkeypatch):
    monkeypatch.setenv("TRUST_PROXY_HEADERS", "1")
    assert rate_limit.client_key(FakeRequest(host="9.9.9.9")) == "9.9.9.9"


# --- enforcement -----------------------------------------------------------


def test_requests_within_the_limit_succeed(limited_client):
    for _ in range(3):
        assert limited_client.get("/limited").status_code == 200


def test_exceeding_the_limit_returns_429(limited_client):
    for _ in range(3):
        limited_client.get("/limited")
    assert limited_client.get("/limited").status_code == 429


def test_a_throttled_response_says_when_to_retry(limited_client):
    for _ in range(4):
        response = limited_client.get("/limited")
    assert response.status_code == 429
    assert int(response.headers["Retry-After"]) == 60
    assert "Rate limit exceeded" in response.json()["detail"]


def test_remaining_budget_is_reported(limited_client):
    first = limited_client.get("/limited")
    assert int(first.headers["x-ratelimit-remaining"]) == 2


def test_exempt_routes_are_never_throttled(limited_client):
    for _ in range(10):
        limited_client.get("/limited")
    # Throttling a liveness probe would restart a container that is merely busy.
    assert limited_client.get("/probe").status_code == 200


def test_nothing_is_throttled_when_limiting_is_off(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT", "off")
    client = build_app("off")
    for _ in range(20):
        assert client.get("/limited").status_code == 200


# --- the real app ----------------------------------------------------------


@pytest.mark.parametrize(
    "route", ["api.main.health", "api.main.ready", "api.main.models"]
)
def test_the_service_exempts_its_probes(route):
    # slowapi records exemptions by "module.function" name, and does so at
    # import time regardless of whether limiting is enabled - which is why this
    # works with the suite's RATE_LIMIT=off. Reaching into _exempt_routes is
    # the only way to assert the registration without rebuilding the real app;
    # the enforcement itself is covered behaviourally above.
    import api.main as api_main

    assert route in api_main.limiter._exempt_routes


def test_prediction_endpoints_are_not_exempt():
    import api.main as api_main

    assert "api.main.predict" not in api_main.limiter._exempt_routes
    assert "api.main.predict_batch" not in api_main.limiter._exempt_routes


def test_retry_after_falls_back_when_the_window_is_unreadable():
    class Odd:
        limit = type("L", (), {"limit": object()})()

    assert rate_limit.retry_after_seconds(Odd()) == 60


# --- a throttled request is still traceable --------------------------------


def test_a_throttled_response_still_carries_a_request_id(limited_client):
    for _ in range(4):
        response = limited_client.get("/limited")
    assert response.status_code == 429
    # The rate limiter used to sit outside the request-id middleware, so the
    # requests most worth investigating were the ones that left no trace.
    assert response.headers[REQUEST_ID_HEADER]


def test_the_service_puts_request_ids_outside_rate_limiting():
    """Order matters: Starlette runs the most recently added middleware first."""
    from slowapi.middleware import SlowAPIMiddleware as SlowAPI

    import api.main as api_main

    classes = [m.cls for m in api_main.app.user_middleware]
    names = [c.__name__ for c in classes]
    assert SlowAPI in classes
    # The request-id middleware is registered via app.middleware("http"), which
    # Starlette wraps in BaseHTTPMiddleware; it must come first (outermost).
    assert names.index("BaseHTTPMiddleware") < names.index("SlowAPIMiddleware")
