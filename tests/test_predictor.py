"""The predictor is the shared path for the UI, the API and evaluation."""

import numpy as np
import pytest

from src.config import LABELS, MODEL_KEYS, NUM_LABELS
from src.inference import predictor as predictor_module
from src.inference.predictor import (
    ModelUnavailableError,
    Prediction,
    available_models,
    check_artifacts,
    is_lfs_pointer,
    load_predictor,
    model_status,
)
from src.utils.preprocessing import preprocess_tweet
from tests.conftest import requires_model

# --- LFS pointer detection -------------------------------------------------


def test_detects_an_lfs_pointer(lfs_pointer):
    assert is_lfs_pointer(lfs_pointer) is True


def test_real_file_is_not_a_pointer(real_artifact):
    assert is_lfs_pointer(real_artifact) is False


def test_missing_file_is_not_a_pointer(tmp_path):
    assert is_lfs_pointer(tmp_path / "nothing-here.bin") is False


def test_directory_is_not_a_pointer(tmp_path):
    assert is_lfs_pointer(tmp_path) is False


# --- availability ----------------------------------------------------------


def test_available_models_is_a_subset_of_known_models():
    assert set(available_models()) <= set(MODEL_KEYS)


@pytest.mark.parametrize("model", MODEL_KEYS)
def test_model_status_reports_every_model(model):
    status = model_status()[model]
    assert set(status) == {
        "display_name",
        "path",
        "available",
        "reason",
        "version",
        "loaded",
        "version_on_disk",
        "stale",
        "artifacts",
    }
    assert status["available"] is (status["reason"] is None)


def test_pointer_stub_is_reported_with_a_fix(monkeypatch, lfs_pointer):
    monkeypatch.setitem(predictor_module.REQUIRED_ARTIFACTS, "lstm", (lfs_pointer,))
    reason = check_artifacts("lstm")
    assert "Git LFS pointer" in reason
    assert "git lfs pull" in reason
    assert "src.training.train_lstm" in reason


def test_missing_artifact_is_reported_with_a_fix(monkeypatch, tmp_path):
    monkeypatch.setitem(
        predictor_module.REQUIRED_ARTIFACTS, "gru", (tmp_path / "absent.keras",)
    )
    reason = check_artifacts("gru")
    assert "is missing" in reason
    assert "src.training.train_gru" in reason


def test_unavailable_model_is_excluded_from_available_models(monkeypatch, lfs_pointer):
    monkeypatch.setitem(predictor_module.REQUIRED_ARTIFACTS, "lstm", (lfs_pointer,))
    assert "lstm" not in available_models()


def test_loading_a_stubbed_model_raises_model_unavailable(monkeypatch, lfs_pointer):
    monkeypatch.setitem(predictor_module.REQUIRED_ARTIFACTS, "lstm", (lfs_pointer,))
    with pytest.raises(ModelUnavailableError, match="Git LFS pointer"):
        load_predictor("lstm")


def test_model_unavailable_is_a_runtime_error():
    # The API relies on this to tell "not loaded here" apart from "broken".
    assert issubclass(ModelUnavailableError, RuntimeError)


@pytest.mark.parametrize("name", ["", "bogus", "LR", "bert2"])
def test_unknown_model_names_are_rejected(name):
    with pytest.raises(KeyError):
        load_predictor(name)


# --- probability ordering --------------------------------------------------


@pytest.mark.parametrize(
    "classes, expected",
    [
        ([0, 1, 2], [0.1, 0.7, 0.2]),
        ([2, 0, 1], [0.7, 0.2, 0.1]),
        ([1, 2, 0], [0.2, 0.1, 0.7]),
    ],
)
def test_probability_columns_follow_label_index_not_column_order(classes, expected):
    probs = np.array([[0.1, 0.7, 0.2]])
    ordered = predictor_module._order_by_label_index(probs, classes)
    assert ordered.shape == (1, NUM_LABELS)
    assert np.allclose(ordered[0], expected)


# --- predictions against real artifacts ------------------------------------


@requires_model("lr")
def test_predict_returns_a_known_label():
    prediction = load_predictor("lr").predict("i absolutely love this")
    assert isinstance(prediction, Prediction)
    assert prediction.label in LABELS
    assert prediction.model == "lr"


@requires_model("lr")
def test_probabilities_are_a_distribution_over_the_labels():
    prediction = load_predictor("lr").predict("this is terrible")
    assert set(prediction.probabilities) == set(LABELS)
    assert pytest.approx(sum(prediction.probabilities.values()), abs=1e-5) == 1.0


@requires_model("lr")
def test_confidence_is_the_largest_probability():
    prediction = load_predictor("lr").predict("the meeting is at noon")
    assert prediction.confidence == pytest.approx(max(prediction.probabilities.values()))
    assert prediction.probabilities[prediction.label] == pytest.approx(
        prediction.confidence
    )


@requires_model("lr")
def test_predict_proba_shape_matches_the_input():
    probs = load_predictor("lr").predict_proba(["good", "bad", "fine"])
    assert probs.shape == (3, NUM_LABELS)


@requires_model("lr")
def test_empty_input_returns_an_empty_matrix():
    assert load_predictor("lr").predict_proba([]).shape == (0, NUM_LABELS)


@requires_model("lr")
def test_predictors_are_cached():
    assert load_predictor("lr") is load_predictor("lr")


@requires_model("lr")
def test_raw_text_is_preprocessed_before_prediction():
    # Preprocessing is idempotent, so predicting on raw text must give exactly
    # what predicting on already-preprocessed text gives.
    raw = "  @someone THIS is GREAT   "
    predictor = load_predictor("lr")
    assert predictor.predict(raw).probabilities == (
        predictor.predict(preprocess_tweet(raw)).probabilities
    )
