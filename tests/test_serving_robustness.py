"""Regressions for issues found in review.

Each test here corresponds to a defect that shipped: a version the service was
not actually serving, a model built once per concurrent request, a CLI that
crashed on a fresh clone, and input that looked like text but was not.
"""

import threading

import pytest

from src.config import MODEL_KEYS
from src.inference import predictor as predictor_module
from tests.conftest import requires_model

pytest.importorskip("fastapi")


# --- the reported version must be the one being served ---------------------


@requires_model("lr")
def test_models_reports_the_loaded_version_not_the_one_on_disk(monkeypatch):
    from src.inference.predictor import load_predictor, model_status

    predictor = load_predictor("lr")
    monkeypatch.setattr(predictor, "version", "deadbeefcafe")

    entry = model_status()["lr"]
    assert entry["version"] == "deadbeefcafe"
    assert entry["loaded"] is True
    assert entry["stale"] is True
    assert entry["version_on_disk"] != "deadbeefcafe"


@pytest.mark.parametrize("model", MODEL_KEYS)
def test_an_unloaded_model_is_not_reported_as_stale(model):
    from src.inference.predictor import model_status

    entry = model_status()[model]
    if not entry["loaded"]:
        assert entry["stale"] is False


def test_unavailable_models_have_no_version(monkeypatch, lfs_pointer):
    from src.inference.predictor import model_status

    monkeypatch.setitem(predictor_module.REQUIRED_ARTIFACTS, "lstm", (lfs_pointer,))
    entry = model_status()["lstm"]
    assert entry["available"] is False
    assert entry["version"] is None
    assert entry["artifacts"] == {}


# --- one model per key, however many threads ask at once -------------------


def test_concurrent_first_loads_build_the_model_once(monkeypatch):
    """Four simultaneous requests used to build four models.

    `functools.cache` locks its bookkeeping, not the call, so every thread that
    misses runs the loader. For the transformer that is a ~2 GB spike.
    """
    built = []

    class Slow:
        def __init__(self):
            built.append(1)
            threading.Event().wait(0.05)

    monkeypatch.setitem(predictor_module._PREDICTORS, "lstm", Slow)
    predictor_module.clear_cache()

    results = []
    threads = [
        threading.Thread(
            target=lambda: results.append(predictor_module.load_predictor("lstm"))
        )
        for _ in range(8)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    predictor_module.clear_cache()
    assert len(built) == 1
    assert all(r is results[0] for r in results)


def test_an_alias_shares_the_cached_instance(monkeypatch):
    built = []

    class Counting:
        def __init__(self):
            built.append(1)

    monkeypatch.setitem(predictor_module._PREDICTORS, "roberta", Counting)
    predictor_module.clear_cache()

    first = predictor_module.load_predictor("bert")
    second = predictor_module.load_predictor("roberta")
    predictor_module.clear_cache()

    assert first is second
    assert len(built) == 1


# --- input that is not text ------------------------------------------------


@pytest.mark.parametrize("text", ["   ", "\t", "\n\n", " \u00a0 "])
def test_whitespace_only_text_is_rejected(api_client, text):
    # min_length counts characters, so these used to be scored and returned a
    # confident-looking label for nothing.
    assert api_client.post("/predict", json={"text": text}).status_code == 422


def test_whitespace_only_items_are_rejected_in_a_batch(api_client):
    response = api_client.post("/predict/batch", json={"texts": ["fine", "   "]})
    assert response.status_code == 422


# --- calibration on a fresh clone ------------------------------------------


def test_calibrate_skips_models_whose_weights_are_absent(monkeypatch, lfs_pointer):
    from src import calibration

    monkeypatch.setitem(predictor_module.REQUIRED_ARTIFACTS, "lstm", (lfs_pointer,))
    # Must not raise: `--models lstm` on a fresh clone is normal, not an error.
    assert calibration.calibrate(["lstm"]) == []


def test_calibration_survives_a_null_temperature(monkeypatch, tmp_path):
    from src import calibration
    from src.utils.io import save_json

    monkeypatch.setitem(calibration.MODEL_DIRS, "lr", tmp_path)
    save_json({"temperature": None}, tmp_path / "calibration.json")
    # float(None) raises TypeError, which used to escape and 500 every request.
    assert calibration.load_temperature("lr") is None
