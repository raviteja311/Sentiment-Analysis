"""The API contract, including how it degrades when a model is not loadable."""

import pytest

from src.config import LABELS, MODEL_KEYS
from src.inference.predictor import ModelUnavailableError, available_models
from tests.conftest import requires_model

pytest.importorskip("fastapi")


# --- probes ----------------------------------------------------------------


def test_health_is_ok_even_without_models(api_client):
    response = api_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_lists_available_models(api_client):
    body = api_client.get("/health").json()
    assert set(body["models_available"]) <= set(MODEL_KEYS)


def test_models_endpoint_describes_every_model(api_client):
    body = api_client.get("/models").json()["models"]
    assert set(body) == set(MODEL_KEYS)
    for info in body.values():
        assert info["available"] is (info["reason"] is None)


@pytest.mark.skipif(not available_models(), reason="no usable model artifacts")
def test_ready_is_200_when_a_model_can_be_served(api_client):
    response = api_client.get("/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_ready_is_503_when_nothing_can_be_served(api_client, monkeypatch):
    import api.main as api_main

    monkeypatch.setattr(api_main, "available_models", lambda: [])
    response = api_client.get("/ready")
    assert response.status_code == 503
    assert "git lfs" in response.json()["detail"].lower()


# --- prediction ------------------------------------------------------------


@requires_model("lr")
def test_predict_returns_a_label_and_a_distribution(api_client):
    response = api_client.post(
        "/predict", json={"text": "this is fantastic", "model": "lr"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "lr"
    assert body["label"] in LABELS
    assert set(body["probabilities"]) == set(LABELS)
    assert pytest.approx(sum(body["probabilities"].values()), abs=1e-5) == 1.0
    assert 0.0 <= body["confidence"] <= 1.0


@requires_model("lr")
def test_predict_defaults_to_the_lr_model(api_client):
    body = api_client.post("/predict", json={"text": "good"}).json()
    assert body["model"] == "lr"


@requires_model("lr")
def test_batch_predict_returns_one_prediction_per_text(api_client):
    response = api_client.post(
        "/predict/batch", json={"texts": ["great", "awful", "fine"], "model": "lr"}
    )
    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 3


# --- validation ------------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        {"text": "hello", "model": "bogus"},
        {"text": "", "model": "lr"},
        {"model": "lr"},
        {"text": "x" * 5_001, "model": "lr"},
    ],
)
def test_bad_requests_are_rejected_with_422(api_client, payload):
    assert api_client.post("/predict", json=payload).status_code == 422


@pytest.mark.parametrize(
    "payload",
    [
        {"texts": [], "model": "lr"},
        {"texts": ["x"] * 257, "model": "lr"},
        {"texts": ["x"], "model": "bogus"},
    ],
)
def test_bad_batch_requests_are_rejected_with_422(api_client, payload):
    assert api_client.post("/predict/batch", json=payload).status_code == 422


# --- degradation -----------------------------------------------------------


def test_unloadable_model_answers_503_not_500(api_client, monkeypatch):
    import api.main as api_main

    def unavailable(model):
        raise ModelUnavailableError(
            "models/lstm/model_final.keras is a Git LFS pointer, not real model weights."
        )

    monkeypatch.setattr(api_main, "load_predictor", unavailable)
    response = api_client.post("/predict", json={"text": "hello", "model": "lstm"})
    assert response.status_code == 503
    assert "Git LFS pointer" in response.json()["detail"]


def test_unloadable_model_also_degrades_on_the_batch_endpoint(api_client, monkeypatch):
    import api.main as api_main

    monkeypatch.setattr(
        api_main,
        "load_predictor",
        lambda model: (_ for _ in ()).throw(ModelUnavailableError("missing")),
    )
    response = api_client.post("/predict/batch", json={"texts": ["hi"], "model": "gru"})
    assert response.status_code == 503
