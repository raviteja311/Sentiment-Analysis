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


def test_unknown_model_names_leave_no_lock_behind():
    # Validation used to happen inside the per-model lock, so every unknown name
    # ever asked for created a permanent entry in the lock table.
    before = set(predictor_module._LOAD_LOCKS)
    with pytest.raises(KeyError):
        load_predictor("definitely-not-a-model")
    assert set(predictor_module._LOAD_LOCKS) == before


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


# --- serving performance ----------------------------------------------------


def _stub_predictor(forward):
    """A Predictor whose forward pass is `forward`, with no artifacts needed."""
    stub = predictor_module.Predictor.__new__(predictor_module.Predictor)
    stub.key = "lr"
    stub.temperature = 1.0
    stub.version = None
    stub._predict_proba = forward
    return stub


@pytest.mark.parametrize(
    "value, expected",
    [(None, None), ("", None), ("4", 4), ("0", None), ("-2", None), ("many", None)],
)
def test_the_inference_cap_is_read_from_the_environment(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("MAX_CONCURRENT_INFERENCE", raising=False)
    else:
        monkeypatch.setenv("MAX_CONCURRENT_INFERENCE", value)
    assert predictor_module.inference_limit() == expected


def test_no_cap_means_no_semaphore():
    assert predictor_module.build_inference_slots(None) is None
    assert predictor_module.build_inference_slots(2) is not None


def test_the_cap_bounds_simultaneous_forward_passes(monkeypatch):
    import threading

    monkeypatch.setattr(
        predictor_module, "_INFERENCE_SLOTS", predictor_module.build_inference_slots(1)
    )
    inside = 0
    peak = 0
    guard = threading.Lock()

    def forward(cleaned):
        nonlocal inside, peak
        with guard:
            inside += 1
            peak = max(peak, inside)
        threading.Event().wait(0.02)
        with guard:
            inside -= 1
        return np.full((len(cleaned), 3), 1 / 3, dtype=np.float32)

    stub = _stub_predictor(forward)
    threads = [
        threading.Thread(target=stub.predict_proba, args=(["x"],)) for _ in range(6)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert peak == 1


def test_without_a_cap_forward_passes_run_concurrently(monkeypatch):
    import threading

    monkeypatch.setattr(predictor_module, "_INFERENCE_SLOTS", None)
    # Every call has to be inside the forward pass at once for the barrier to
    # open; with a cap of one this would time out instead.
    barrier = threading.Barrier(3, timeout=5)

    def forward(cleaned):
        barrier.wait()
        return np.full((len(cleaned), 3), 1 / 3, dtype=np.float32)

    stub = _stub_predictor(forward)
    threads = [
        threading.Thread(target=stub.predict_proba, args=(["x"],)) for _ in range(3)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert barrier.broken is False


class _FakeTorch:
    def __init__(self):
        self.calls = []

    def set_num_threads(self, n):
        self.calls.append(n)


@pytest.mark.parametrize(
    "value, calls", [(None, []), ("", []), ("3", [3]), ("0", []), ("lots", [])]
)
def test_torch_threads_follow_the_environment(monkeypatch, value, calls):
    if value is None:
        monkeypatch.delenv("TORCH_NUM_THREADS", raising=False)
    else:
        monkeypatch.setenv("TORCH_NUM_THREADS", value)
    torch = _FakeTorch()
    predictor_module.configure_torch_threads(torch)
    assert torch.calls == calls


@requires_model("lstm")
def test_the_direct_call_matches_model_predict(monkeypatch):
    pytest.importorskip("tensorflow")
    texts = ["what a great day", "this is awful", "the meeting is at noon", "!!!"]
    predictor = predictor_module.load_predictor("lstm")

    fast = predictor.predict_proba(texts)
    monkeypatch.setattr(predictor_module, "FAST_PATH_MAX_TEXTS", 0)
    slow = predictor.predict_proba(texts)

    np.testing.assert_allclose(fast, slow, atol=1e-5)
    assert fast.shape == (len(texts), 3)


@requires_model("lstm")
def test_large_batches_still_go_through_model_predict(monkeypatch):
    pytest.importorskip("tensorflow")
    predictor = predictor_module.load_predictor("lstm")
    calls = []
    original = predictor._model.predict

    def counting(*args, **kwargs):
        calls.append(len(args[0]))
        return original(*args, **kwargs)

    monkeypatch.setattr(predictor._model, "predict", counting)
    predictor.predict_proba(["fine"] * (predictor_module.FAST_PATH_MAX_TEXTS + 1))
    assert calls == [predictor_module.FAST_PATH_MAX_TEXTS + 1]
    predictor.predict_proba(["fine"] * 2)
    assert calls == [predictor_module.FAST_PATH_MAX_TEXTS + 1]


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
