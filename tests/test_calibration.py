"""Temperature scaling.

The behavioural claims worth pinning down are that calibration never changes a
prediction, and that fitting on one sample genuinely improves calibration on
another - not just on the data it was fitted to.
"""

import numpy as np
import pytest

from src import calibration
from src.config import MODEL_KEYS

pytest.importorskip("scipy")


@pytest.fixture
def overconfident():
    """A model that is 95% confident and 70% correct - the real failure mode."""
    rng = np.random.default_rng(0)
    n = 2_000
    labels = rng.integers(0, 3, size=n)
    probs = np.full((n, 3), 0.025)
    correct = rng.random(n) < 0.70
    for i in range(n):
        predicted = labels[i] if correct[i] else (labels[i] + 1) % 3
        probs[i, predicted] = 0.95
    return probs, labels


# --- applying a temperature ------------------------------------------------


def test_temperature_of_one_changes_nothing(overconfident):
    probs, _ = overconfident
    assert np.allclose(calibration.apply_temperature(probs, 1.0), probs)


def test_output_is_still_a_distribution(overconfident):
    probs, _ = overconfident
    scaled = calibration.apply_temperature(probs, 2.5)
    assert np.allclose(scaled.sum(axis=1), 1.0)
    assert (scaled >= 0).all()


def test_high_temperature_softens_confidence(overconfident):
    probs, _ = overconfident
    softened = calibration.apply_temperature(probs, 3.0)
    assert softened.max(axis=1).mean() < probs.max(axis=1).mean()


def test_low_temperature_sharpens_confidence(overconfident):
    probs, _ = overconfident
    sharpened = calibration.apply_temperature(probs, 0.5)
    assert sharpened.max(axis=1).mean() > probs.max(axis=1).mean()


@pytest.mark.parametrize("temperature", [0.25, 0.5, 1.0, 2.0, 5.0])
def test_predictions_never_change(overconfident, temperature):
    # The property that lets calibration ship without re-verifying every metric.
    probs, _ = overconfident
    scaled = calibration.apply_temperature(probs, temperature)
    assert (scaled.argmax(axis=1) == probs.argmax(axis=1)).all()


def test_zero_probabilities_do_not_produce_nan():
    probs = np.array([[1.0, 0.0, 0.0]])
    scaled = calibration.apply_temperature(probs, 2.0)
    assert np.isfinite(scaled).all()


# --- measures --------------------------------------------------------------


def test_perfect_calibration_scores_zero_error():
    # Always 100% confident and always right.
    probs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    labels = np.array([0, 1, 2])
    assert calibration.expected_calibration_error(probs, labels) == pytest.approx(0.0)


def test_overconfident_model_has_calibration_error(overconfident):
    probs, labels = overconfident
    # ~95% confident, ~70% accurate: the gap should be roughly a quarter.
    assert calibration.expected_calibration_error(probs, labels) > 0.2


def test_brier_and_nll_reward_the_truth():
    confident_right = np.array([[0.99, 0.005, 0.005]])
    confident_wrong = np.array([[0.005, 0.99, 0.005]])
    labels = np.array([0])
    assert calibration.brier_score(confident_right, labels) < calibration.brier_score(
        confident_wrong, labels
    )
    assert calibration.negative_log_likelihood(
        confident_right, labels
    ) < calibration.negative_log_likelihood(confident_wrong, labels)


def test_measure_reports_every_statistic(overconfident):
    probs, labels = overconfident
    assert set(calibration.measure(probs, labels)) == {
        "ece",
        "nll",
        "brier",
        "mean_confidence",
        "accuracy",
    }


# --- fitting ---------------------------------------------------------------


def test_fitting_softens_an_overconfident_model(overconfident):
    probs, labels = overconfident
    assert calibration.fit_temperature(probs, labels) > 1.0


def test_fitting_improves_calibration_on_unseen_data(overconfident):
    probs, labels = overconfident
    fit, held_out = slice(0, 1_000), slice(1_000, 2_000)

    temperature = calibration.fit_temperature(probs[fit], labels[fit])

    before = calibration.expected_calibration_error(probs[held_out], labels[held_out])
    after = calibration.expected_calibration_error(
        calibration.apply_temperature(probs[held_out], temperature), labels[held_out]
    )
    assert after < before


def test_fitting_leaves_an_already_calibrated_model_alone():
    rng = np.random.default_rng(1)
    n = 2_000
    labels = rng.integers(0, 3, size=n)
    # 70% confident and 70% correct.
    probs = np.full((n, 3), 0.15)
    correct = rng.random(n) < 0.70
    for i in range(n):
        predicted = labels[i] if correct[i] else (labels[i] + 1) % 3
        probs[i, predicted] = 0.70

    assert calibration.fit_temperature(probs, labels) == pytest.approx(1.0, abs=0.25)


def test_held_out_improvement_is_positive_for_an_overconfident_model(overconfident):
    probs, labels = overconfident
    assert calibration.held_out_improvement(probs, labels) > 0


def test_held_out_improvement_is_negligible_for_a_calibrated_model():
    rng = np.random.default_rng(2)
    n = 2_000
    labels = rng.integers(0, 3, size=n)
    probs = np.full((n, 3), 0.15)
    correct = rng.random(n) < 0.70
    for i in range(n):
        predicted = labels[i] if correct[i] else (labels[i] + 1) % 3
        probs[i, predicted] = 0.70

    # Nothing to gain: whatever the sign, the magnitude should be tiny, which is
    # what the adoption gate keys off.
    assert abs(calibration.held_out_improvement(probs, labels)) < 0.02


# --- storage and wiring ----------------------------------------------------


@pytest.mark.parametrize("model", MODEL_KEYS)
def test_calibration_path_sits_beside_the_model(model):
    from src.config import MODEL_DIRS

    assert calibration.calibration_path(model).parent == MODEL_DIRS[model]


def test_missing_calibration_file_means_no_temperature(monkeypatch, tmp_path):
    monkeypatch.setitem(calibration.MODEL_DIRS, "lr", tmp_path)
    assert calibration.load_temperature("lr") is None


def test_unreadable_calibration_file_is_ignored(monkeypatch, tmp_path):
    monkeypatch.setitem(calibration.MODEL_DIRS, "lr", tmp_path)
    (tmp_path / "calibration.json").write_text("{}", encoding="utf-8")
    assert calibration.load_temperature("lr") is None


def test_temperature_round_trips(monkeypatch, tmp_path):
    from src.utils.io import save_json

    monkeypatch.setitem(calibration.MODEL_DIRS, "lr", tmp_path)
    save_json({"temperature": 1.75}, tmp_path / "calibration.json")
    assert calibration.load_temperature("lr") == pytest.approx(1.75)


def test_table_reports_before_and_after():
    record = {
        "model": "bert",
        "temperature": 1.8,
        "before": {"ece": 0.2, "mean_confidence": 0.95, "accuracy": 0.7},
        "after": {"ece": 0.02, "mean_confidence": 0.72, "accuracy": 0.7},
    }
    table = calibration.markdown_table([record])
    assert "bert" in table
    assert "1.80" in table


def test_table_says_so_when_nothing_is_calibrated():
    assert "No models calibrated yet" in calibration.markdown_table([])
