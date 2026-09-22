"""Central configuration for the sentiment analysis project.

Before this module existed, each training script carried its own copy of the
hyperparameters, the Streamlit app carried its own copy of the label list, and
nothing connected either of them to the artifacts sitting in ``models/``.
Anything that more than one entry point has to agree on belongs here.

The module deliberately imports nothing beyond the standard library, so it stays
cheap to import from tests, the API and the UI alike.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
METRICS_DIR = REPORTS_DIR / "metrics"

LR_DIR = MODELS_DIR / "lr"
LSTM_DIR = MODELS_DIR / "lstm"
GRU_DIR = MODELS_DIR / "gru"
BERT_DIR = MODELS_DIR / "bert"

# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

# Index order is fixed by the tweet_eval sentiment dataset and by every trained
# artifact under models/. Do not reorder it. models/bert/config.json carries the
# same mapping (it once held HuggingFace's LABEL_0/1/2 placeholders); a test
# asserts the two agree, so a change here without a retrain will fail the suite.
LABELS: tuple[str, ...] = ("negative", "neutral", "positive")
NUM_LABELS = len(LABELS)
ID2LABEL: dict[int, str] = {i: name for i, name in enumerate(LABELS)}
LABEL2ID: dict[str, int] = {name: i for i, name in enumerate(LABELS)}

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

DATASET_NAME = "cardiffnlp/tweet_eval"
DATASET_CONFIG = "sentiment"
SPLITS: tuple[str, ...] = ("train", "validation", "test")

SEED = 42

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

MODEL_KEYS: tuple[str, ...] = ("lr", "lstm", "gru", "bert")

MODEL_DISPLAY_NAMES: dict[str, str] = {
    "lr": "Logistic Regression",
    "lstm": "Bi-LSTM",
    "gru": "Bi-GRU",
    # The checkpoint is RoBERTa (models/bert/config.json reports model_type
    # "roberta"); the directory is called "bert" for historical reasons.
    "bert": "Twitter-RoBERTa",
}

# Files that must exist, and must not be Git LFS pointer stubs, before a model
# can be considered usable. Consumed by src.inference.predictor.
REQUIRED_ARTIFACTS: dict[str, tuple[Path, ...]] = {
    "lr": (LR_DIR / "pipeline.joblib",),
    "lstm": (LSTM_DIR / "model_final.keras", LSTM_DIR / "tokenizer.joblib"),
    "gru": (GRU_DIR / "model_final.keras", GRU_DIR / "tokenizer.joblib"),
    "bert": (
        BERT_DIR / "model.safetensors",
        BERT_DIR / "config.json",
        BERT_DIR / "tokenizer.json",
    ),
}

MODEL_DIRS: dict[str, Path] = {
    "lr": LR_DIR,
    "lstm": LSTM_DIR,
    "gru": GRU_DIR,
    "bert": BERT_DIR,
}

# ---------------------------------------------------------------------------
# Remote weights
# ---------------------------------------------------------------------------
#
# The large weights live on the Hugging Face Hub rather than in git. Half a
# gigabyte of safetensors in Git LFS costs bandwidth on every clone, fork and CI
# checkout, and leaves pointer stubs behind when anyone forgets `git lfs pull`.
# Everything small enough to be unremarkable in git - the LR pipeline, the
# tokenizers, the transformer's config and vocabulary - stays in git, so a plain
# clone still has a working model.

DEFAULT_MODELS_REPO = "RAVITEJA311/sentiment-analysis-models"
MODELS_REPO = os.environ.get("SENTIMENT_MODELS_REPO", DEFAULT_MODELS_REPO)
MODELS_REPO_REVISION = os.environ.get("SENTIMENT_MODELS_REVISION", "main")

# Files fetched from the Hub, relative to MODELS_DIR. The layout on the Hub
# mirrors models/ exactly.
REMOTE_ARTIFACTS: dict[str, tuple[str, ...]] = {
    "lr": (),
    "lstm": ("lstm/model_final.keras", "lstm/best.keras"),
    "gru": ("gru/model_final.keras", "gru/best.keras"),
    "bert": ("bert/model.safetensors", "bert/training_args.bin"),
}

# Shown in the error message when an artifact is missing or is an LFS stub.
RETRAIN_COMMANDS: dict[str, str] = {
    "lr": "python -m src.training.train_lr",
    "lstm": "python -m src.training.train_lstm",
    "gru": "python -m src.training.train_gru",
    "bert": "python -m src.training.train_bert",
}


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
#
# The values below are the ones the committed artifacts were actually trained
# with, verified against the artifacts themselves rather than copied from the
# prose documentation (which listed max_features=5000, solver=liblinear and
# max_iter=200 for the LR model - none of which match the fitted estimator).


@dataclass(frozen=True)
class LRConfig:
    """TF-IDF + LogisticRegression baseline."""

    max_features: int = 10_000
    ngram_range: tuple[int, int] = (1, 2)
    min_df: int = 3
    max_df: float = 0.9
    C: float = 1.0
    solver: str = "saga"
    max_iter: int = 2_000
    class_weight: str = "balanced"


@dataclass(frozen=True)
class SequenceConfig:
    """Shared settings for the Bi-LSTM and Bi-GRU models."""

    max_vocab: int = 10_000
    max_len: int = 80
    embed_dim: int = 100
    batch_size: int = 64
    epochs: int = 4
    oov_token: str = "<OOV>"
    early_stopping_patience: int = 2
    # Off by default, on evidence. The training split is 45% neutral and 16%
    # negative, and the linear baseline uses balanced weights, so weighting
    # these models looked obviously right - but measured, it trades a 2-3 point
    # gain on negative for a 6-10 point collapse on neutral, and costs both
    # models accuracy and macro F1. See MODEL_CARD.md. Set to "balanced" to
    # train weighted anyway; the training run records which was used.
    class_weight: str | None = None


@dataclass(frozen=True)
class BertConfig:
    """Fine-tuning settings for the Twitter-RoBERTa checkpoint."""

    base_model: str = "cardiffnlp/twitter-roberta-base-sentiment"
    max_len: int = 128
    batch_size: int = 8
    # The committed run used 3 epochs. Its own trainer_state.json shows epoch 1
    # was the best checkpoint (macro F1 0.7806) while validation loss climbed
    # from 0.5146 to 0.9767 by epoch 3, so any retrain should stop earlier.
    epochs: int = 2
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    early_stopping_patience: int = 1


LR_CONFIG = LRConfig()
SEQUENCE_CONFIG = SequenceConfig()
BERT_CONFIG = BertConfig()
