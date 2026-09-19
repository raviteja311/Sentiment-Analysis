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


def test_table_sorts_by_model(sample_report):
    other = dict(sample_report, model="gru", display_name="Bi-GRU")
    rows = evaluate.markdown_table([sample_report, other]).splitlines()
    assert "Bi-GRU" in rows[2]
    assert "Logistic Regression" in rows[3]


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
