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


# --- the untouched base checkpoint ------------------------------------------
#
# The transformer's base checkpoint is already fine-tuned on TweetEval, so its
# score is the bar our fine-tuning has to clear. It is evaluated, never served.


def test_the_baseline_is_not_a_servable_model():
    from src.config import BASELINE_KEY, MODEL_KEYS
    from src.inference.predictor import check_artifacts, load_predictor

    assert BASELINE_KEY not in MODEL_KEYS
    with pytest.raises(KeyError):
        check_artifacts(BASELINE_KEY)
    with pytest.raises(KeyError):
        load_predictor(BASELINE_KEY)


def test_the_api_refuses_the_baseline(api_client):
    from src.config import BASELINE_KEY

    response = api_client.post("/predict", json={"text": "hi", "model": BASELINE_KEY})
    assert response.status_code == 422


class _StubBaseline:
    key = "roberta_base"

    def __init__(self, base_model=None):
        self.base_model = base_model

    def predict_proba(self, texts):
        import numpy as np

        return np.tile([0.1, 0.2, 0.7], (len(texts), 1))


def test_include_base_scores_the_baseline_and_writes_its_record(monkeypatch, tmp_path):
    from src.config import BASELINE_DISPLAY_NAME, BASELINE_KEY, ROBERTA_CONFIG

    monkeypatch.setattr(evaluate, "METRICS_DIR", tmp_path)
    monkeypatch.setattr(evaluate, "baseline_predictor", _StubBaseline)
    monkeypatch.setattr(evaluate, "available_models", lambda: [])
    monkeypatch.setattr(
        evaluate,
        "load_raw_dataset",
        lambda: {"test": {"text": ["a", "b"], "label": [2, 0]}},
    )

    reports = evaluate.evaluate_models(None, include_base=True)

    assert [report["model"] for report in reports] == [BASELINE_KEY]
    record = load_json(tmp_path / f"{BASELINE_KEY}.json")
    assert record["display_name"] == BASELINE_DISPLAY_NAME
    assert record["hyperparameters"] == {
        "base_model": ROBERTA_CONFIG.base_model,
        "preprocessing": "cardiff-v1",
        "fine_tuned": False,
    }
    assert record["metrics"]["test"]["accuracy"] == 0.5


def test_the_baseline_is_off_by_default(monkeypatch):
    monkeypatch.setattr(evaluate, "available_models", lambda: [])
    touched = []
    monkeypatch.setattr(evaluate, "load_raw_dataset", lambda: touched.append(1))
    assert evaluate.evaluate_models(None) == []
    assert touched == []


def test_an_unreachable_hub_skips_the_baseline_rather_than_crashing(monkeypatch, caplog):
    def offline(base_model=None):
        raise OSError("no network")

    monkeypatch.setattr(evaluate, "baseline_predictor", offline)
    monkeypatch.setattr(evaluate, "available_models", lambda: [])
    monkeypatch.setattr(
        evaluate, "load_raw_dataset", lambda: {"test": {"text": ["a"], "label": [0]}}
    )
    assert evaluate.evaluate_models(None, include_base=True) == []
    assert "Skipping roberta_base" in caplog.text


def test_the_cli_exposes_include_base(monkeypatch):
    seen = {}

    def fake(models, split, batch_size, include_base, base_model):
        seen["include_base"] = include_base
        return []

    monkeypatch.setattr(evaluate, "evaluate_models", fake)
    evaluate.main(["--include-base"])
    assert seen == {"include_base": True}


def test_the_table_names_the_baseline_as_not_fine_tuned(monkeypatch, tmp_path):
    from src.config import BASELINE_DISPLAY_NAME, BASELINE_KEY

    monkeypatch.setattr(evaluate, "METRICS_DIR", tmp_path)
    record = build_report(
        model=BASELINE_KEY,
        hyperparameters={},
        dataset={},
        metrics={"test": compute_metrics([0, 1, 2], [0, 1, 2])},
        display_name=BASELINE_DISPLAY_NAME,
    )
    save_json(record, tmp_path / f"{BASELINE_KEY}.json")

    # Included without being asked for, once its record exists.
    reports = evaluate.load_reports()
    assert [r["model"] for r in reports] == [BASELINE_KEY]
    assert BASELINE_DISPLAY_NAME in evaluate.markdown_table(reports)


def test_the_baseline_predictor_loads_the_base_checkpoint_with_its_own_preprocessing(
    monkeypatch,
):
    transformers = pytest.importorskip("transformers")
    from src.config import BASELINE_DISPLAY_NAME, BASELINE_KEY, ROBERTA_CONFIG

    calls = []

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, name, **kwargs):
            calls.append(("tokenizer", name))
            return cls()

    class FakeModel:
        @classmethod
        def from_pretrained(cls, name, **kwargs):
            calls.append(("model", name))
            return cls()

        def to(self, device):
            return self

        def eval(self):
            return self

    monkeypatch.setattr(transformers, "AutoTokenizer", FakeTokenizer)
    monkeypatch.setattr(transformers, "AutoModelForSequenceClassification", FakeModel)

    predictor = evaluate.BaselinePredictor(device="cpu")

    base = ROBERTA_CONFIG.base_model
    assert calls == [("tokenizer", base), ("model", base)]
    assert predictor.key == BASELINE_KEY
    assert predictor.display_name == BASELINE_DISPLAY_NAME
    assert predictor.preprocessing == "cardiff-v1"
    assert predictor.temperature == 1.0
    assert predictor.version == base


# --- an alternative base checkpoint -------------------------------------------


def test_the_configured_base_keeps_the_plain_key_and_others_get_a_suffix():
    from src.config import ALTERNATIVE_ROBERTA_BASE, BASELINE_KEY, ROBERTA_CONFIG

    assert evaluate.baseline_key() == BASELINE_KEY
    assert evaluate.baseline_key(ROBERTA_CONFIG.base_model) == BASELINE_KEY
    alternative = evaluate.baseline_key(ALTERNATIVE_ROBERTA_BASE)
    assert alternative == f"{BASELINE_KEY}__twitter_roberta_base_sentiment_latest"
    assert evaluate.is_baseline(alternative) and evaluate.is_baseline(BASELINE_KEY)
    assert not evaluate.is_baseline("roberta")
    assert ALTERNATIVE_ROBERTA_BASE in evaluate.baseline_display_name(
        ALTERNATIVE_ROBERTA_BASE
    )


def test_an_alternative_base_is_scored_into_its_own_record(monkeypatch, tmp_path):
    from src.config import ALTERNATIVE_ROBERTA_BASE, BASELINE_KEY

    monkeypatch.setattr(evaluate, "METRICS_DIR", tmp_path)
    monkeypatch.setattr(evaluate, "available_models", lambda: [])
    monkeypatch.setattr(
        evaluate,
        "load_raw_dataset",
        lambda: {"test": {"text": ["a", "b"], "label": [2, 0]}},
    )
    seen = []

    def fake_baseline(base_model=None):
        seen.append(base_model)
        return _StubBaseline()

    monkeypatch.setattr(evaluate, "baseline_predictor", fake_baseline)

    reports = evaluate.evaluate_models(
        None, include_base=True, base_model=ALTERNATIVE_ROBERTA_BASE
    )

    assert seen == [ALTERNATIVE_ROBERTA_BASE]
    key = evaluate.baseline_key(ALTERNATIVE_ROBERTA_BASE)
    assert [r["model"] for r in reports] == [key]
    record = load_json(tmp_path / f"{key}.json")
    assert record["hyperparameters"]["base_model"] == ALTERNATIVE_ROBERTA_BASE
    assert record["hyperparameters"]["fine_tuned"] is False
    assert ALTERNATIVE_ROBERTA_BASE in record["display_name"]
    # The default base's record is untouched.
    assert not (tmp_path / f"{BASELINE_KEY}.json").exists()


def test_the_table_picks_up_every_baseline_record(monkeypatch, tmp_path):
    from src.config import ALTERNATIVE_ROBERTA_BASE, BASELINE_KEY

    monkeypatch.setattr(evaluate, "METRICS_DIR", tmp_path)
    for key, base in (
        (BASELINE_KEY, None),
        (evaluate.baseline_key(ALTERNATIVE_ROBERTA_BASE), ALTERNATIVE_ROBERTA_BASE),
    ):
        record = build_report(
            model=key,
            hyperparameters=evaluate.baseline_hyperparameters(base),
            dataset={},
            metrics={"validation": compute_metrics([0, 1, 2], [0, 1, 2])},
            display_name=evaluate.baseline_display_name(base),
        )
        save_json(record, tmp_path / f"{key}.json")

    reports = evaluate.load_reports()
    assert len(reports) == 2
    assert ALTERNATIVE_ROBERTA_BASE in evaluate.markdown_table(reports)


def test_the_cli_passes_the_base_model_through(monkeypatch):
    from src.config import ALTERNATIVE_ROBERTA_BASE

    seen = {}

    def fake(models, split, batch_size, include_base, base_model):
        seen.update(include_base=include_base, base_model=base_model)
        return []

    monkeypatch.setattr(evaluate, "evaluate_models", fake)
    evaluate.main(["--include-base", "--base-model", ALTERNATIVE_ROBERTA_BASE])
    assert seen == {"include_base": True, "base_model": ALTERNATIVE_ROBERTA_BASE}


def test_the_baseline_predictor_loads_the_checkpoint_it_is_given(monkeypatch):
    transformers = pytest.importorskip("transformers")
    from src.config import ALTERNATIVE_ROBERTA_BASE

    calls = []

    class Fake:
        @classmethod
        def from_pretrained(cls, name, **kwargs):
            calls.append(name)
            return cls()

        def to(self, device):
            return self

        def eval(self):
            return self

    monkeypatch.setattr(transformers, "AutoTokenizer", Fake)
    monkeypatch.setattr(transformers, "AutoModelForSequenceClassification", Fake)

    predictor = evaluate.BaselinePredictor(ALTERNATIVE_ROBERTA_BASE, device="cpu")
    assert calls == [ALTERNATIVE_ROBERTA_BASE] * 2
    assert predictor.key == evaluate.baseline_key(ALTERNATIVE_ROBERTA_BASE)
    assert predictor.version == ALTERNATIVE_ROBERTA_BASE
    assert predictor.preprocessing == "cardiff-v1"


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
