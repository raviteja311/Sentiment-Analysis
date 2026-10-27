"""Sampling test errors for hand-tagging, and counting the tags."""

import csv

import numpy as np
import pytest

from src import errors, evaluate


@pytest.fixture
def scratch(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate, "PREDICTIONS_DIR", tmp_path / "predictions")
    monkeypatch.setattr(errors, "ERRORS_PATH", tmp_path / "errors.csv")
    return tmp_path


def test_sample_holds_only_errors_and_is_reproducible():
    labels = np.array([0, 1, 2, 0, 1, 2, 0, 1])
    predictions = np.array([0, 2, 2, 1, 1, 0, 0, 0])
    chosen = errors.sample_errors(labels, predictions, size=3)
    assert set(chosen) <= {1, 3, 5, 7}
    assert len(chosen) == 3
    assert chosen.tolist() == errors.sample_errors(labels, predictions, size=3).tolist()


def test_sample_is_capped_at_the_number_of_errors():
    assert len(errors.sample_errors([0, 1], [1, 1], size=100)) == 1


def test_export_writes_untagged_rows_with_text(scratch):
    labels = [0, 1, 2, 2]
    probabilities = np.eye(3)[[0, 2, 2, 1]] * 0.9 + 0.1 / 3
    evaluate.save_predictions("roberta", "test", labels, probabilities)
    dataset = {"test": {"text": ["a", "b", "c", "d"], "label": labels}}

    assert errors.export(dataset=dataset) == 2

    with errors.ERRORS_PATH.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["text"] for row in rows] == ["b", "d"]
    assert rows[0]["gold"] == "neutral" and rows[0]["predicted"] == "positive"
    assert all(row["tag"] == "" for row in rows)


def _write(path, tags):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=errors.COLUMNS)
        writer.writeheader()
        for i, tag in enumerate(tags):
            writer.writerow(
                dict.fromkeys(errors.COLUMNS, "")
                | {"index": i, "gold": "neutral", "predicted": "positive", "tag": tag}
            )


def test_summary_counts_tags_and_reports_untagged_rows(tmp_path):
    path = tmp_path / "errors.csv"
    _write(path, ["sarcasm", "sarcasm", "negation", ""])
    summary = errors.summarise(path)
    assert summary["tags"] == {"sarcasm": 2, "negation": 1}
    assert summary["untagged"] == ["3"]
    assert "| sarcasm | 2 | 67% |" in errors.markdown_summary(summary)


def test_summary_rejects_tags_outside_the_list(tmp_path):
    path = tmp_path / "errors.csv"
    _write(path, ["sarcasm", "Sarcasm!"])
    with pytest.raises(ValueError, match="unknown tags"):
        errors.summarise(path)
