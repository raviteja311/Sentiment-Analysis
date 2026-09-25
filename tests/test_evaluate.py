"""Metrics records and the Markdown table that gets pasted into the README."""

import pytest

from src import evaluate
from src.config import LABELS
from src.utils.io import load_json, save_json
from src.utils.metrics import build_report, compute_metrics
from tests.conftest import requires_model


@pytest.fixture
def sample_report():
    metrics = compute_metrics([0, 1, 2, 0, 1, 2], [0, 1, 2, 0, 1, 1])
    return build_report(
        model="lr",
        hyperparameters={"evaluation_batch_size": 64},
        dataset={"name": "demo", "split": "test", "size": 6},
        metrics={"test": metrics},
    )


# --- metrics ---------------------------------------------------------------


def test_perfect_predictions_score_one():
    metrics = compute_metrics([0, 1, 2], [0, 1, 2])
    assert metrics["accuracy"] == 1.0
    assert metrics["f1_macro"] == 1.0


def test_metrics_include_macro_recall():
    # TweetEval's official sentiment metric. Two of three classes fully
    # recalled and the third (label 2, both predicted as 1) not at all.
    metrics = compute_metrics([0, 1, 2, 2], [0, 1, 1, 1])
    assert metrics["recall_macro"] == pytest.approx((1 + 1 + 0) / 3)


def test_table_has_a_macro_recall_column(sample_report):
    header = evaluate.markdown_table([sample_report]).splitlines()[0]
    assert "Macro recall" in header
    assert f"{sample_report['metrics']['test']['recall_macro']:.4f}" in (
        evaluate.markdown_table([sample_report])
    )


def test_table_shows_n_a_for_records_without_macro_recall(sample_report):
    # Every committed record predates the metric; the table must still render.
    old = dict(sample_report)
    old["metrics"] = {"test": dict(sample_report["metrics"]["test"])}
    del old["metrics"]["test"]["recall_macro"]
    table = evaluate.markdown_table([old])
    assert "n/a" in table.splitlines()[2]


def test_metrics_include_per_class_and_confusion_matrix():
    metrics = compute_metrics([0, 1, 2], [0, 1, 2])
    assert len(metrics["f1_per_class"]) == len(LABELS)
    assert metrics["confusion_matrix"] == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


def test_confusion_matrix_records_the_actual_mistake():
    metrics = compute_metrics([0, 0], [0, 1])
    assert metrics["confusion_matrix"][0] == [1, 1, 0]
    assert metrics["accuracy"] == 0.5


# --- report records --------------------------------------------------------


def test_report_carries_hyperparameters_and_dataset(sample_report):
    assert set(sample_report) == {
        "model",
        "display_name",
        "created_at",
        "hyperparameters",
        "dataset",
        "metrics",
    }
    assert sample_report["display_name"] == "Logistic Regression"
    assert sample_report["dataset"]["size"] == 6


def test_report_is_json_round_trippable(sample_report, tmp_path):
    path = tmp_path / "lr.json"
    save_json(sample_report, path)
    assert load_json(path) == sample_report


def test_save_json_creates_missing_directories(tmp_path):
    path = tmp_path / "reports" / "metrics" / "lr.json"
    save_json({"a": 1}, path)
    assert path.is_file()


def test_load_reports_reads_only_what_exists(tmp_path, monkeypatch, sample_report):
    monkeypatch.setattr(evaluate, "METRICS_DIR", tmp_path)
    assert evaluate.load_reports() == []
    save_json(sample_report, tmp_path / "lr.json")
    assert [report["model"] for report in evaluate.load_reports()] == ["lr"]


# --- evaluation merges into the training record ---------------------------


def test_evaluation_keeps_the_hyperparameters_training_recorded(
    tmp_path, monkeypatch, sample_report
):
    monkeypatch.setattr(evaluate, "METRICS_DIR", tmp_path)
    training_record = dict(
        sample_report, hyperparameters={"max_features": 10000, "solver": "saga"}
    )
    save_json(training_record, tmp_path / "lr.json")

    merged = evaluate.update_record(
        "lr", "test", compute_metrics([0, 1], [0, 1]), [0, 1], 64
    )

    assert merged["hyperparameters"] == {"max_features": 10000, "solver": "saga"}
    assert merged["metrics"]["test"]["accuracy"] == 1.0
    assert merged["evaluation"]["size"] == 2


def test_evaluation_does_not_discard_other_splits(tmp_path, monkeypatch, sample_report):
    monkeypatch.setattr(evaluate, "METRICS_DIR", tmp_path)
    training_record = dict(sample_report)
    training_record["metrics"] = {
        "validation": compute_metrics([0, 1], [0, 1]),
        "test": compute_metrics([0, 1], [0, 0]),
    }
    save_json(training_record, tmp_path / "lr.json")

    merged = evaluate.update_record(
        "lr", "test", compute_metrics([0, 1], [0, 1]), [0, 1], 64
    )

    assert set(merged["metrics"]) == {"validation", "test"}
    assert merged["metrics"]["test"]["accuracy"] == 1.0


def test_evaluation_without_a_training_record_claims_no_hyperparameters(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(evaluate, "METRICS_DIR", tmp_path)

    record = evaluate.update_record(
        "gru", "test", compute_metrics([0, 1], [0, 1]), [0, 1], 32
    )

    assert record["hyperparameters"] == {}
    assert record["evaluation"]["batch_size"] == 32


def test_example_count_prefers_the_evaluated_size():
    report = {
        "evaluation": {"split": "test", "size": 12284},
        "dataset": {"split_sizes": {"test": 999}},
    }
    assert evaluate.example_count(report, "test") == 12284


def test_example_count_falls_back_to_the_training_split_sizes():
    report = {"dataset": {"split_sizes": {"test": 12284, "validation": 2000}}}
    assert evaluate.example_count(report, "validation") == 2000


def test_example_count_is_dash_when_nothing_records_it():
    assert evaluate.example_count({"dataset": {}}, "test") == "-"


# --- markdown table --------------------------------------------------------


def test_table_says_so_when_there_is_nothing_to_report():
    assert "No metrics recorded yet" in evaluate.markdown_table([])


def test_table_has_a_column_per_label(sample_report):
    header = evaluate.markdown_table([sample_report]).splitlines()[0]
    for label in LABELS:
        assert f"F1 {label}" in header


def test_table_renders_one_row_per_split(sample_report):
    rows = evaluate.markdown_table([sample_report]).splitlines()
    assert len(rows) == 3  # header, divider, one split
    assert "Logistic Regression" in rows[2]


def test_table_puts_the_best_model_first(sample_report):
    weaker = dict(
        sample_report,
        model="gru",
        display_name="Bi-GRU",
        metrics={"test": compute_metrics([0, 1, 2], [0, 0, 0])},
    )
    rows = evaluate.markdown_table([weaker, sample_report]).splitlines()
    assert "Logistic Regression" in rows[2]
    assert "Bi-GRU" in rows[3]


def test_table_formats_scores_to_four_decimals(sample_report):
    assert "0.8333" in evaluate.markdown_table([sample_report])


# --- scoring a real artifact ----------------------------------------------


@requires_model("lr")
def test_evaluate_predictions_scores_a_real_model():
    texts = [
        "i love this, it is wonderful",
        "this is terrible and awful",
        "the meeting is at noon",
    ]
    metrics = evaluate.evaluate_predictions("lr", texts, [2, 0, 1])
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert len(metrics["confusion_matrix"]) == len(LABELS)
