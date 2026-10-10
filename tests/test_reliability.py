"""Reliability diagrams and the cross-model temperature comparison."""

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from src import evaluate, reliability
from src.calibration import expected_calibration_error
from src.utils.io import save_json
from src.utils.metrics import build_report, compute_metrics


def _overconfident(n, seed):
    """Predictions that claim ~0.95 while being right about 70% of the time."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 3, n)
    predicted = np.where(rng.random(n) < 0.7, labels, (labels + 1) % 3)
    probs = np.full((n, 3), 0.025)
    probs[np.arange(n), predicted] = 0.95
    return labels, probs


@pytest.fixture
def scratch(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate, "PREDICTIONS_DIR", tmp_path / "predictions")
    monkeypatch.setattr(evaluate, "METRICS_DIR", tmp_path / "metrics")
    monkeypatch.setattr(reliability, "COMPARISON_PATH", tmp_path / "comparison.json")
    monkeypatch.setattr(reliability, "FIGURES_DIR", tmp_path / "figures")
    return tmp_path


def test_bins_reproduce_the_expected_calibration_error():
    labels, probs = _overconfident(3000, seed=0)
    bins = reliability.reliability_bins(probs, labels)
    total = sum(b["count"] for b in bins)
    ece = sum(b["count"] / total * abs(b["accuracy"] - b["confidence"]) for b in bins)
    assert total == len(labels)
    assert ece == pytest.approx(expected_calibration_error(probs, labels))


def test_an_overconfident_model_gets_softened_on_test(scratch):
    for split, seed in (("validation", 1), ("test", 2)):
        labels, probs = _overconfident(2000, seed)
        evaluate.save_predictions("lstm", split, labels, probs, temperature=1.0)
    save_json(
        build_report("lstm", {}, {}, {"test": compute_metrics([0, 1], [0, 1])}),
        evaluate.METRICS_DIR / "lstm.json",
    )

    (record,) = reliability.run()

    assert record["fitted_temperature"] > 1.5
    assert record["test"]["ece_after"] < record["test"]["ece_before"]
    assert record["test"]["mean_confidence_after"] < 0.95
    svg = (scratch / "figures" / "reliability_lstm.svg").read_text(encoding="utf-8")
    ET.fromstring(svg)  # well-formed
    assert "After temperature scaling" in svg
    assert "Fitted T" in reliability.markdown_table([record])


def test_diagram_omits_the_calibrated_series_when_nothing_was_softened():
    record = {
        "display_name": "Logistic <Regression>",
        "fitted_temperature": 1.0,
        "test": {"ece_before": 0.04, "ece_after": 0.04},
        "bins_before": [{"confidence": 0.6, "accuracy": 0.6, "count": 10}],
        "bins_after": [{"confidence": 0.6, "accuracy": 0.6, "count": 10}],
    }
    svg = reliability.reliability_svg(record)
    ET.fromstring(svg)  # the name was escaped
    assert "After temperature scaling" not in svg
    assert "no softening" in svg
