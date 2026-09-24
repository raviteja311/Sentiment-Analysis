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
