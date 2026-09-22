"""Structured logging and request correlation."""

import io
import json
import logging

import pytest

from api import observability
from tests.conftest import requires_model

pytest.importorskip("fastapi")


@pytest.fixture
def log_output():
    """Capture real formatted log lines, not just records.

    A handler of our own rather than caplog: api.main calls configure_logging()
    on import, which replaces the root handlers and would drop caplog's.
    """
    buffer = io.StringIO()
    handler = logging.StreamHandler(buffer)
    handler.addFilter(observability.RequestIdFilter())
    handler.setFormatter(observability.JsonFormatter())

    root = logging.getLogger()
    previous_handlers, previous_level = root.handlers, root.level
    root.handlers = [handler]
    root.setLevel(logging.INFO)
    try:
        yield buffer
    finally:
        root.handlers, root.level = previous_handlers, previous_level


def lines(buffer) -> list[dict]:
    return [json.loads(line) for line in buffer.getvalue().splitlines() if line.strip()]


# --- request ids -----------------------------------------------------------


def test_an_id_is_generated_when_the_caller_sends_none():
    assert observability.normalise_request_id(None)


def test_a_caller_supplied_id_is_kept():
    assert observability.normalise_request_id("trace-abc.123") == "trace-abc.123"


@pytest.mark.parametrize(
    "value",
    [
        "has space",
        "newline\ninjected",
        'quote"injected',
        "{}".format("x" * 65),
        "",
        "semi;colon",
    ],
)
def test_unsafe_ids_are_replaced(value):
    # An inbound id lands in a JSON log line; a newline in it would forge one.
    assert observability.normalise_request_id(value) != value


def test_generated_ids_are_unique():
    assert observability.new_request_id() != observability.new_request_id()


# --- formatting ------------------------------------------------------------


def test_each_line_is_a_json_object(log_output):
    logging.getLogger("test").info("hello")
    assert lines(log_output)[0]["message"] == "hello"


def test_lines_carry_the_standard_fields(log_output):
    logging.getLogger("test").info("hello")
    entry = lines(log_output)[0]
    assert set(entry) >= {"timestamp", "level", "logger", "message", "request_id"}
    assert entry["level"] == "INFO"
    assert entry["logger"] == "test"


def test_extra_fields_are_included(log_output):
    logging.getLogger("test").info("event", extra={"model": "lr", "duration_ms": 12.5})
    entry = lines(log_output)[0]
    assert entry["model"] == "lr"
    assert entry["duration_ms"] == 12.5


def test_exceptions_are_captured(log_output):
    try:
        raise ValueError("boom")
    except ValueError:
        logging.getLogger("test").exception("failed")
    assert "ValueError: boom" in lines(log_output)[0]["exception"]


def test_unserialisable_values_do_not_break_the_line(log_output):
    logging.getLogger("test").info("event", extra={"obj": object()})
    assert lines(log_output)[0]["obj"]


def test_the_current_request_id_is_attached(log_output):
    observability.set_request_id("req-42")
    try:
        logging.getLogger("test").info("hello")
    finally:
        observability.set_request_id(None)
    assert lines(log_output)[0]["request_id"] == "req-42"


# --- middleware ------------------------------------------------------------


def test_responses_carry_a_request_id(api_client):
    response = api_client.get("/health")
    assert response.headers[observability.REQUEST_ID_HEADER]


def test_a_caller_supplied_id_is_echoed_back(api_client):
    response = api_client.get(
        "/health", headers={observability.REQUEST_ID_HEADER: "trace-123"}
    )
    assert response.headers[observability.REQUEST_ID_HEADER] == "trace-123"


def test_an_unsafe_id_is_not_echoed_back(api_client):
    response = api_client.get(
        "/health", headers={observability.REQUEST_ID_HEADER: "a b c"}
    )
    assert response.headers[observability.REQUEST_ID_HEADER] != "a b c"


def test_each_request_gets_its_own_id(api_client):
    first = api_client.get("/health").headers[observability.REQUEST_ID_HEADER]
    second = api_client.get("/health").headers[observability.REQUEST_ID_HEADER]
    assert first != second


def test_requests_are_logged_with_status_and_duration(api_client, log_output):
    api_client.get("/health")
    access = [entry for entry in lines(log_output) if entry["logger"] == "api.access"]
    assert access
    assert access[0]["status"] == 200
    assert access[0]["path"] == "/health"
    assert isinstance(access[0]["duration_ms"], float)


# --- what must not be logged -----------------------------------------------


@requires_model("lr")
def test_the_classified_text_is_never_logged(api_client, log_output):
    secret = "my private message about something personal"
    api_client.post("/predict", json={"text": secret, "model": "lr"})

    assert secret not in log_output.getvalue()
    for word in ("private", "personal"):
        assert word not in log_output.getvalue()


@requires_model("lr")
def test_predictions_log_the_model_version_and_length(api_client, log_output):
    api_client.post("/predict", json={"text": "this is fantastic", "model": "lr"})
    predictions = [
        entry for entry in lines(log_output) if entry["message"] == "prediction"
    ]
    assert predictions
    entry = predictions[0]
    assert entry["model"] == "lr"
    assert entry["model_version"]
    assert entry["text_length"] == len("this is fantastic")
    assert entry["request_id"]
