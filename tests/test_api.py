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
    # The message must name the command that actually fixes it; the weights
    # moved off Git LFS to the Hub, so `git lfs pull` no longer would.
    assert "make fetch-weights" in response.json()["detail"]


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


# --- deprecated aliases ----------------------------------------------------


@requires_model("roberta")
def test_the_old_model_key_still_works(api_client):
    response = api_client.post("/predict", json={"text": "great", "model": "bert"})
    assert response.status_code == 200
    # The response reports the canonical name, so callers migrate naturally.
    assert response.json()["model"] == "roberta"


@requires_model("roberta")
def test_the_alias_and_the_canonical_key_agree(api_client):
    old = api_client.post("/predict", json={"text": "great", "model": "bert"}).json()
    new = api_client.post("/predict", json={"text": "great", "model": "roberta"}).json()
    assert old == new


def test_unknown_models_are_still_rejected(api_client):
    response = api_client.post("/predict", json={"text": "hi", "model": "gpt"})
    assert response.status_code == 422


@requires_model("roberta")
def test_the_alias_works_on_the_batch_endpoint(api_client):
    response = api_client.post(
        "/predict/batch", json={"texts": ["great", "awful"], "model": "bert"}
    )
    assert response.status_code == 200
    assert all(p["model"] == "roberta" for p in response.json()["predictions"])


# --- batch requests validate their items -----------------------------------


@pytest.mark.parametrize(
    "texts",
    [
        ["fine", ""],  # an empty string is not a text
        ["fine", "x" * 5_001],  # nor is one past the single-text limit
        [""],
    ],
)
def test_batch_items_are_validated_individually(api_client, texts):
    # Length limits on the list alone bound how many texts arrive, not how long
    # each one is, so these used to slip through.
    response = api_client.post("/predict/batch", json={"texts": texts, "model": "lr"})
    assert response.status_code == 422


@requires_model("lr")
def test_batch_runs_one_forward_pass_for_the_whole_batch(api_client, monkeypatch):
    from src.inference.predictor import load_predictor

    predictor = load_predictor("lr")
    calls = []
    original = predictor.predict_proba

    def counting(texts):
        calls.append(len(texts))
        return original(texts)

    monkeypatch.setattr(predictor, "predict_proba", counting)
    api_client.post("/predict/batch", json={"texts": ["a", "b", "c", "d"], "model": "lr"})
    assert calls == [4]


# --- warm-up ----------------------------------------------------------------
#
# The lifespan handler only runs when the TestClient is used as a context
# manager, which is why these tests build their own client rather than using
# the api_client fixture.


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, []),
        ("", []),
        ("lr", ["lr"]),
        ("lr, roberta", ["lr", "roberta"]),
        ("ALL", ["lr", "lstm", "gru", "roberta"]),
    ],
)
def test_preload_list_is_read_from_the_environment(monkeypatch, value, expected):
    import api.main as api_main

    if value is None:
        monkeypatch.delenv("PRELOAD_MODELS", raising=False)
    else:
        monkeypatch.setenv("PRELOAD_MODELS", value)
    assert api_main.preload_list() == expected


@requires_model("lr")
def test_preloaded_models_are_loaded_before_the_first_request(monkeypatch):
    from fastapi.testclient import TestClient

    from api.main import app

    monkeypatch.setenv("PRELOAD_MODELS", "lr")
    with TestClient(app) as client:
        status = client.get("/models").json()["models"]
    assert status["lr"]["loaded"] is True


def test_nothing_is_preloaded_by_default(monkeypatch):
    from fastapi.testclient import TestClient

    from api.main import app
    from src.inference.predictor import loaded_models

    monkeypatch.delenv("PRELOAD_MODELS", raising=False)
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
    assert loaded_models() == {}


def test_an_unknown_or_unavailable_preload_name_does_not_stop_startup(
    monkeypatch, lfs_pointer
):
    from fastapi.testclient import TestClient

    from api.main import app
    from src.inference import predictor as predictor_module

    monkeypatch.setitem(predictor_module.REQUIRED_ARTIFACTS, "gru", (lfs_pointer,))
    monkeypatch.setenv("PRELOAD_MODELS", "nope,gru")
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
    assert "gru" not in predictor_module.loaded_models()


def test_a_loader_that_raises_does_not_stop_startup(monkeypatch):
    from fastapi.testclient import TestClient

    from api.main import app
    from src.inference import predictor as predictor_module

    def broken():
        raise RuntimeError("corrupt archive")

    monkeypatch.setitem(predictor_module._PREDICTORS, "lstm", broken)
    monkeypatch.setenv("PRELOAD_MODELS", "lstm")
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
    assert "lstm" not in predictor_module.loaded_models()
