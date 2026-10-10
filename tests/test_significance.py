"""Bootstrap intervals and McNemar's test over saved predictions."""

import numpy as np
import pytest
from sklearn.metrics import f1_score, recall_score

from src import evaluate, significance
from src.config import PROJECT_ROOT
from src.utils.io import save_json
from src.utils.metrics import build_report, compute_metrics


@pytest.fixture
def scratch(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate, "PREDICTIONS_DIR", tmp_path / "predictions")
    monkeypatch.setattr(evaluate, "METRICS_DIR", tmp_path / "metrics")
    monkeypatch.setattr(significance, "METRICS_DIR", tmp_path / "metrics")
    monkeypatch.setattr(significance, "SIGNIFICANCE_PATH", tmp_path / "sig.json")
    return tmp_path


def _one_hot(predictions):
    return np.eye(3)[predictions]


def _noisy_predictions(labels, accuracy, seed):
    rng = np.random.default_rng(seed)
    wrong = rng.random(len(labels)) > accuracy
    return np.where(wrong, (labels + rng.integers(1, 3, len(labels))) % 3, labels)


def _save_model(model, labels, predictions):
    evaluate.save_predictions(model, "test", labels, _one_hot(predictions))
    record = build_report(
        model=model,
        hyperparameters={},
        dataset={},
        metrics={"test": compute_metrics(labels, predictions)},
    )
    save_json(record, evaluate.METRICS_DIR / f"{model}.json")


# --- the scores themselves ----------------------------------------------------


def test_macro_scores_match_sklearn():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 3, 500)
    predictions = _noisy_predictions(labels, 0.6, seed=1)

    matrix = significance.confusion_matrices(labels, predictions, [slice(None)])
    scores = significance.macro_scores(matrix)

    assert scores["f1_macro"][0] == pytest.approx(
        f1_score(labels, predictions, average="macro")
    )
    assert scores["recall_macro"][0] == pytest.approx(
        recall_score(labels, predictions, average="macro")
    )


def test_resamples_are_reproducible():
    first = significance.resample_indices(100, n_resamples=5, seed=7)
    again = significance.resample_indices(100, n_resamples=5, seed=7)
    np.testing.assert_array_equal(first, again)


def test_interval_brackets_the_point_estimate_and_narrows_with_more_data():
    widths = []
    for n in (300, 3000):
        labels = np.random.default_rng(n).integers(0, 3, n)
        predictions = _noisy_predictions(labels, 0.7, seed=n)
        samples = significance.resample_indices(n, n_resamples=300)
        low, high = significance.bootstrap_intervals(labels, predictions, samples)[
            "f1_macro"
        ]
        assert low <= f1_score(labels, predictions, average="macro") <= high
        widths.append(high - low)
    assert widths[1] < widths[0]


# --- McNemar --------------------------------------------------------------


def test_mcnemar_counts_only_the_discordant_tweets():
    labels = np.array([0, 0, 0, 0, 1, 1])
    a = np.array([0, 0, 0, 1, 1, 0])  # right on 0,1,2,4
    b = np.array([0, 1, 1, 1, 1, 1])  # right on 0,4,5
    result = significance.mcnemar(labels, a, b)
    assert result["only_a_correct"] == 2
    assert result["only_b_correct"] == 1


def test_identical_models_are_not_significantly_different():
    labels = np.array([0, 1, 2, 1])
    assert significance.mcnemar(labels, labels, labels)["p_value"] == 1.0


def test_a_lopsided_disagreement_is_significant():
    labels = np.zeros(200, dtype=int)
    a = np.zeros(200, dtype=int)
    b = np.r_[np.ones(60, dtype=int), np.zeros(140, dtype=int)]
    assert significance.mcnemar(labels, a, b)["p_value"] < 1e-6


# --- comparisons over saved predictions -------------------------------------


def test_compare_reports_a_minus_b(scratch):
    labels = np.random.default_rng(3).integers(0, 3, 400)
    better = _noisy_predictions(labels, 0.8, seed=4)
    worse = _noisy_predictions(labels, 0.6, seed=5)
    _save_model("lstm", labels, better)
    _save_model("gru", labels, worse)

    result = significance.compare("lstm", "gru", samples=significance.resample_indices(400, 200))

    expected = f1_score(labels, better, average="macro") - f1_score(
        labels, worse, average="macro"
    )
    difference = result["difference"]["f1_macro"]
    assert difference["estimate"] == pytest.approx(expected)
    assert difference["interval"][0] > 0
    assert result["mcnemar"]["p_value"] < 0.001


def test_compare_refuses_models_scored_on_different_tweets(scratch):
    _save_model("lstm", np.array([0, 1, 2]), np.array([0, 1, 2]))
    _save_model("gru", np.array([0, 1, 1]), np.array([0, 1, 1]))
    with pytest.raises(ValueError, match="same tweets"):
        significance.compare("lstm", "gru")


def test_run_records_intervals_and_comparisons(scratch):
    labels = np.random.default_rng(8).integers(0, 3, 300)
    _save_model("lstm", labels, _noisy_predictions(labels, 0.7, seed=9))
    _save_model("gru", labels, _noisy_predictions(labels, 0.7, seed=10))

    result = significance.run(pairs=[("lstm", "gru")])

    assert [(c["a"], c["b"]) for c in result["comparisons"]] == [("lstm", "gru")]
    reports = evaluate.load_reports()
    assert all("bootstrap" in r["metrics"]["test"] for r in reports)
    assert "95% CI" in significance.interval_table(reports)
    assert "McNemar p" in significance.comparison_table(result)


# --- the published tables -----------------------------------------------------


@pytest.mark.parametrize("document", ["README.md", "MODEL_CARD.md"])
def test_the_published_uncertainty_tables_are_the_generated_ones(document):
    if not significance.SIGNIFICANCE_PATH.exists():
        pytest.skip("significance has not been run")
    from src.utils.io import load_json

    text = (PROJECT_ROOT / document).read_text(encoding="utf-8")
    assert significance.interval_table(evaluate.load_reports()) in text
    assert significance.comparison_table(load_json(significance.SIGNIFICANCE_PATH)) in text
