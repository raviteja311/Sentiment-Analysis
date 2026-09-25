"""Training-time behaviour that does not require running a training job."""

import numpy as np
import pytest

from src.config import LABELS, LR_CONFIG, LR_DIR, SEED, SEQUENCE_CONFIG
from src.training.train_sequence import compute_class_weights

# --- the linear baseline is built from its config, all of it ----------------


def _lr_pipeline():
    pytest.importorskip("sklearn")
    from src.training.train_lr import build_pipeline

    return build_pipeline()


def test_lr_pipeline_uses_the_configured_token_pattern():
    # The record written next to the model is asdict(LR_CONFIG). A field that
    # sits in the config but is never handed to the estimator makes that record
    # describe a model that was not trained: exactly what happened when the
    # token pattern was added to the config and the vectorizer kept its default.
    tfidf = _lr_pipeline().named_steps["tfidf"]
    assert tfidf.token_pattern == LR_CONFIG.token_pattern


def test_lr_token_pattern_keeps_the_placeholders_whole():
    tfidf = _lr_pipeline().named_steps["tfidf"]
    tokens = tfidf.build_tokenizer()("<user> loved <url> and <email> today")
    assert "<user>" in tokens
    assert "<url>" in tokens
    assert "<email>" in tokens
    assert "user" not in tokens


def test_lr_pipeline_is_seeded():
    # saga shuffles; the model card says every training run is seeded.
    assert _lr_pipeline().named_steps["clf"].random_state == SEED


@pytest.mark.parametrize(
    "field", ["max_features", "ngram_range", "min_df", "max_df", "token_pattern"]
)
def test_lr_vectorizer_matches_config(field):
    tfidf = _lr_pipeline().named_steps["tfidf"]
    assert getattr(tfidf, field) == getattr(LR_CONFIG, field)


@pytest.mark.parametrize("field", ["C", "max_iter", "class_weight", "solver"])
def test_lr_classifier_matches_config(field):
    clf = _lr_pipeline().named_steps["clf"]
    assert getattr(clf, field) == getattr(LR_CONFIG, field)


@pytest.mark.skipif(
    not (LR_DIR / "pipeline.joblib").is_file(), reason="LR pipeline not present"
)
def test_committed_lr_pipeline_was_trained_with_the_configured_token_pattern():
    """The artifact on disk must agree with the config that claims to describe it."""
    import joblib

    tfidf = joblib.load(LR_DIR / "pipeline.joblib").named_steps["tfidf"]
    assert tfidf.token_pattern == LR_CONFIG.token_pattern
    assert "<user>" in tfidf.vocabulary_


# --- the LR training run --------------------------------------------------------


class _FakeSplit(dict):
    """Just enough of a datasets split: column access and a length."""

    def __len__(self):
        return len(self["text"])


def _fake_dataset():
    phrases = {
        0: "awful terrible bad hate worst",
        1: "meeting today schedule update note",
        2: "great love wonderful best happy",
    }
    texts, labels = [], []
    for label, phrase in phrases.items():
        for i in range(8):
            texts.append(f"{phrase} {i}")
            labels.append(label)
    split = _FakeSplit(text=texts, label=labels)
    return {"train": split, "validation": split, "test": split}


def test_lr_training_fits_the_pipeline_once_and_writes_the_record(monkeypatch, tmp_path):
    from src.training import train_lr
    from src.utils import preprocessing

    monkeypatch.setattr(train_lr, "load_raw_dataset", _fake_dataset)
    monkeypatch.setattr(train_lr, "OUT_DIR", tmp_path / "lr")
    monkeypatch.setattr(train_lr, "METRICS_DIR", tmp_path / "metrics")
    # Training also writes preprocessing.json beside the pipeline; keep that
    # out of the real models/ directory.
    monkeypatch.setitem(preprocessing.MODEL_DIRS, "lr", tmp_path / "lr")

    fits = []
    original_fit = train_lr.Pipeline.fit

    def counting_fit(self, *args, **kwargs):
        fits.append(self)
        return original_fit(self, *args, **kwargs)

    monkeypatch.setattr(train_lr.Pipeline, "fit", counting_fit)

    report = train_lr.main()

    # One fit on the training split. The former GridSearchCV over a single
    # point fitted three cross-validation folds plus the final refit.
    assert len(fits) == 1
    assert (tmp_path / "lr" / "pipeline.joblib").is_file()
    assert (tmp_path / "metrics" / "lr.json").is_file()
    assert report["metrics"]["validation"]["accuracy"] == 1.0


def test_lr_training_prepares_splits_the_shared_way(monkeypatch, tmp_path):
    # Every other training script goes through src.data.prepare_split, so the
    # text reaches the vectorizer preprocessed exactly as the other models see
    # it. A local DataFrame detour used to bypass that.
    from src.config import TRAIN_PREPROCESSING
    from src.training import train_lr

    seen = []
    original = train_lr.prepare_split

    def recording(dataset, split, spec):
        seen.append((split, spec))
        return original(dataset, split, spec)

    monkeypatch.setattr(train_lr, "prepare_split", recording)
    monkeypatch.setattr(train_lr, "load_raw_dataset", _fake_dataset)
    monkeypatch.setattr(train_lr, "OUT_DIR", tmp_path / "lr")
    monkeypatch.setattr(train_lr, "METRICS_DIR", tmp_path / "metrics")
    monkeypatch.setitem(
        train_lr.write_artifact_spec.__globals__["MODEL_DIRS"], "lr", tmp_path
    )

    train_lr.main()
    spec = TRAIN_PREPROCESSING["lr"]
    assert seen == [("train", spec), ("validation", spec), ("test", spec)]


def test_lr_training_records_its_preprocessing_spec(monkeypatch, tmp_path):
    import json

    from src.config import TRAIN_PREPROCESSING
    from src.training import train_lr
    from src.utils import preprocessing

    monkeypatch.setattr(train_lr, "load_raw_dataset", _fake_dataset)
    monkeypatch.setattr(train_lr, "OUT_DIR", tmp_path / "lr")
    monkeypatch.setattr(train_lr, "METRICS_DIR", tmp_path / "metrics")
    monkeypatch.setitem(preprocessing.MODEL_DIRS, "lr", tmp_path / "lr")

    report = train_lr.main()

    # Beside the pipeline, for inference; in the record, for provenance.
    spec_file = tmp_path / "lr" / "preprocessing.json"
    assert json.loads(spec_file.read_text(encoding="utf-8")) == {
        "spec": TRAIN_PREPROCESSING["lr"]
    }
    assert report["hyperparameters"]["preprocessing"] == TRAIN_PREPROCESSING["lr"]


def test_prepare_split_applies_the_requested_spec():
    from src.data import prepare_split

    dataset = {"train": _FakeSplit(text=["I can't &amp; won't"], label=[0])}
    v1, labels = prepare_split(dataset, "train", "glove-v1")
    v2, _ = prepare_split(dataset, "train", "glove-v2")
    assert v1 == ["i can't &amp; won't"]
    assert v2 == ["i can not & will not"]
    assert labels.tolist() == [0]


def test_prepare_split_needs_an_explicit_spec():
    # No default, so a training script cannot silently pick up the wrong one.
    from src.data import prepare_split

    with pytest.raises(TypeError):
        prepare_split({"train": _FakeSplit(text=["x"], label=[0])}, "train")


def test_every_model_has_a_training_spec_the_code_knows():
    from src.config import MODEL_KEYS, TRAIN_PREPROCESSING
    from src.utils.preprocessing import SPECS

    assert set(TRAIN_PREPROCESSING) == set(MODEL_KEYS)
    assert set(TRAIN_PREPROCESSING.values()) <= set(SPECS)


# --- the transformer's training history -------------------------------------------


def test_training_history_records_the_checkpoint_relative_to_the_project():
    pytest.importorskip("transformers")
    from src.config import PROJECT_ROOT
    from src.training.train_roberta import training_history

    class State:
        best_metric = 0.78
        best_model_checkpoint = str(PROJECT_ROOT / "models" / "roberta" / "checkpoint-7")
        epoch = 2.0
        global_step = 14
        log_history = [
            {"loss": 0.9, "step": 7},
            {"eval_loss": 0.5, "eval_f1_macro": 0.77, "epoch": 1.0, "step": 7},
            {"eval_loss": 0.6, "eval_f1_macro": 0.78, "epoch": 2.0, "step": 14},
        ]

    history = training_history(State())
    assert history["best_model_checkpoint"] == "models/roberta/checkpoint-7"
    assert [e["step"] for e in history["evaluations"]] == [7, 14]
    assert history["best_metric"] == 0.78


def test_the_committed_training_history_has_no_absolute_path():
    import json

    from src.config import METRICS_DIR

    history = json.loads(
        (METRICS_DIR / "roberta_training_history.json").read_text(encoding="utf-8")
    )
    checkpoint = history["best_model_checkpoint"]
    assert not checkpoint.startswith(("C:", "/")), checkpoint
    assert "\\" not in checkpoint


def test_balanced_weights_favour_the_rare_class():
    # 80% neutral, 10% each negative and positive.
    labels = np.array([1] * 80 + [0] * 10 + [2] * 10)
    weights = compute_class_weights(labels, "balanced")
    assert weights[0] > weights[1]
    assert weights[2] > weights[1]


def test_balanced_weights_are_one_when_classes_are_even():
    labels = np.array([0, 1, 2] * 30)
    weights = compute_class_weights(labels, "balanced")
    assert all(value == pytest.approx(1.0) for value in weights.values())


def test_no_strategy_means_no_weighting():
    assert compute_class_weights(np.array([0, 1, 2]), None) is None


def test_weights_cover_every_label():
    labels = np.array([0] * 5 + [1] * 10 + [2] * 20)
    assert set(compute_class_weights(labels, "balanced")) == set(range(len(LABELS)))


def test_sequence_config_trains_unweighted_by_default():
    # Balanced weights were measured and made both models worse overall; the
    # option stays, the default does not. See MODEL_CARD.md.
    assert SEQUENCE_CONFIG.class_weight is None
