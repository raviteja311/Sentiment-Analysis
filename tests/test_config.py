"""The config is the project's single source of truth, so check it agrees with
the artifacts on disk - that agreement is exactly what drifted before."""

import json

import pytest

from src import config


def test_labels_are_the_three_sentiment_classes():
    assert config.LABELS == ("negative", "neutral", "positive")
    assert config.NUM_LABELS == 3


def test_label_maps_are_inverses():
    assert config.ID2LABEL == {0: "negative", 1: "neutral", 2: "positive"}
    assert config.LABEL2ID == {name: i for i, name in config.ID2LABEL.items()}


@pytest.mark.parametrize("model", config.MODEL_KEYS)
def test_every_model_has_artifacts_a_directory_and_a_retrain_command(model):
    assert config.REQUIRED_ARTIFACTS[model]
    assert model in config.MODEL_DIRS
    assert model in config.RETRAIN_COMMANDS
    assert model in config.MODEL_DISPLAY_NAMES


@pytest.mark.parametrize("model", config.MODEL_KEYS)
def test_artifact_paths_live_under_the_models_directory(model):
    for path in config.REQUIRED_ARTIFACTS[model]:
        assert config.MODELS_DIR in path.parents


def test_project_root_contains_this_repository():
    assert (config.PROJECT_ROOT / "src").is_dir()
    assert (config.PROJECT_ROOT / "README.md").is_file()


def test_metrics_directory_sits_under_reports():
    assert config.METRICS_DIR.parent == config.REPORTS_DIR


def test_lr_config_matches_what_the_committed_pipeline_was_trained_with():
    # The old documentation claimed 5000 / liblinear / 200. The fitted estimator
    # says otherwise, and the config must follow the estimator.
    assert config.LR_CONFIG.max_features == 10_000
    assert config.LR_CONFIG.solver == "saga"
    assert config.LR_CONFIG.max_iter == 2_000


def test_sequence_config_early_stopping_can_actually_fire():
    # Patience 3 with 4 epochs made the callback decorative.
    assert config.SEQUENCE_CONFIG.early_stopping_patience < config.SEQUENCE_CONFIG.epochs


def test_bert_config_points_at_the_roberta_checkpoint():
    assert "roberta" in config.BERT_CONFIG.base_model


@pytest.mark.skipif(
    not (config.BERT_DIR / "config.json").is_file(),
    reason="models/bert/config.json not present",
)
def test_saved_transformer_config_uses_real_label_names():
    saved = json.loads((config.BERT_DIR / "config.json").read_text(encoding="utf-8"))
    assert {int(k): v for k, v in saved["id2label"].items()} == config.ID2LABEL
    assert saved["label2id"] == config.LABEL2ID
