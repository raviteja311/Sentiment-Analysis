"""Central configuration for the sentiment analysis project.

Before this module existed, each training script carried its own copy of the
hyperparameters, the Streamlit app carried its own copy of the label list, and
nothing connected either of them to the artifacts sitting in ``models/``.
Anything that more than one entry point has to agree on belongs here.

The module deliberately imports nothing beyond the standard library, so it stays
cheap to import from tests, the API and the UI alike.
"""

from __future__ import annotations

import json
import logging
import os
import string
from dataclasses import dataclass
from pathlib import Path

LOGGER = logging.getLogger(__name__)

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
ROBERTA_DIR = MODELS_DIR / "roberta"

# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

# Index order is fixed by the tweet_eval sentiment dataset and by every trained
# artifact under models/. Do not reorder it. models/roberta/config.json carries
# the same mapping (it once held HuggingFace's LABEL_0/1/2 placeholders); a test
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

MODEL_KEYS: tuple[str, ...] = ("lr", "lstm", "gru", "roberta")

# The transformer was called "bert" until its directory and key were corrected:
# the checkpoint is RoBERTa (models/roberta/config.json reports model_type
# "roberta"). The old key still resolves, so saved requests and scripts do not
# break on a naming fix.
MODEL_ALIASES: dict[str, str] = {"bert": "roberta"}


def resolve_model(name: str) -> str:
    """Canonical key for a model name, following deprecated aliases."""
    return MODEL_ALIASES.get(name, name)


MODEL_DISPLAY_NAMES: dict[str, str] = {
    "lr": "Logistic Regression",
    "lstm": "Bi-LSTM",
    "gru": "Bi-GRU",
    "roberta": "Twitter-RoBERTa",
}

# Files that must exist, and must not be Git LFS pointer stubs, before a model
# can be considered usable. Consumed by src.inference.predictor.
REQUIRED_ARTIFACTS: dict[str, tuple[Path, ...]] = {
    "lr": (LR_DIR / "pipeline.joblib",),
    "lstm": (LSTM_DIR / "model_final.keras", LSTM_DIR / "tokenizer.joblib"),
    "gru": (GRU_DIR / "model_final.keras", GRU_DIR / "tokenizer.joblib"),
    "roberta": (
        ROBERTA_DIR / "model.safetensors",
        ROBERTA_DIR / "config.json",
        ROBERTA_DIR / "tokenizer.json",
    ),
}

MODEL_DIRS: dict[str, Path] = {
    "lr": LR_DIR,
    "lstm": LSTM_DIR,
    "gru": GRU_DIR,
    "roberta": ROBERTA_DIR,
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

# The tokenizers and calibration files live in git, the weights on the Hub, and
# the two only agree if they come from the same training run. Fetching whatever
# is on the Hub's `main` branch would silently pair a newer publish with an
# older tokenizer. So the Hub commit is pinned here, together with the sha256
# of every fetched file, and both are checked by src/artifacts.py. Publishing
# (scripts/publish_weights.py) rewrites the manifest, so the pin moves with the
# weights and the two are committed together.
REMOTE_MANIFEST = MODELS_DIR / "remote_manifest.json"


def load_remote_manifest(path: Path = REMOTE_MANIFEST) -> dict | None:
    """The committed manifest, or None when it is absent or unreadable."""
    try:
        with open(path, encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, ValueError):
        return None
    return manifest if isinstance(manifest, dict) else None


def resolve_models_revision(env_value: str | None, manifest: dict | None) -> str:
    """Which Hub revision to fetch: the env var, else the manifest's pin, else main.

    Falling back to ``main`` is the old, unpinned behaviour and is only a
    convenience for a checkout with no manifest at all, so it is logged.
    """
    if env_value:
        return env_value
    if manifest and manifest.get("revision"):
        return str(manifest["revision"])
    LOGGER.warning(
        "%s is missing or has no revision; fetching weights from the mutable "
        "'main' branch, which may not match the committed tokenizers.",
        REMOTE_MANIFEST.name,
    )
    return "main"


MODELS_REPO_REVISION = resolve_models_revision(
    os.environ.get("SENTIMENT_MODELS_REVISION"), load_remote_manifest()
)

# Files fetched from the Hub, relative to MODELS_DIR. The layout on the Hub
# mirrors models/ exactly. Only what inference reads is listed: the recurrent
# models' best.keras is the early-stopping checkpoint that model_final.keras
# already supersedes, and the transformer's training_args.bin is a pickle of
# the Trainer's arguments. Both still exist on the Hub; they are just not
# worth 80 MB of download and a pickle load on every fresh clone.
REMOTE_ARTIFACTS: dict[str, tuple[str, ...]] = {
    "lr": (),
    "lstm": ("lstm/model_final.keras",),
    "gru": ("gru/model_final.keras",),
    "roberta": ("roberta/model.safetensors",),
}

# Which preprocessing spec each model is trained with from now on; the specs
# themselves live in src/utils/preprocessing.py. Used only by training. Serving
# reads the spec an artifact records beside its weights, so this can change
# without altering what the shipped models see until they are retrained.
TRAIN_PREPROCESSING: dict[str, str] = {
    "lr": "glove-v2",
    "lstm": "glove-v2",
    "gru": "glove-v2",
    "roberta": "cardiff-v1",
}

# Shown in the error message when an artifact is missing or is an LFS stub.
RETRAIN_COMMANDS: dict[str, str] = {
    "lr": "python -m src.training.train_lr",
    "lstm": "python -m src.training.train_lstm",
    "gru": "python -m src.training.train_gru",
    "roberta": "python -m src.training.train_roberta",
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
    # scikit-learn's default drops the angle brackets, merging <user>
    # into the ordinary noun "user"; this keeps them whole.
    token_pattern: str = r"(?u)<\w+>|\b\w\w+\b"


@dataclass(frozen=True)
class SequenceConfig:
    """Shared settings for the Bi-LSTM and Bi-GRU models."""

    max_vocab: int = 10_000
    max_len: int = 80
    embed_dim: int = 100
    batch_size: int = 64
    epochs: int = 4
    oov_token: str = "<OOV>"
    # Keras's default filters include < and >, which silently turned the
    # placeholders into the plain words "user" and "url" - so a mention was
    # indistinguishable from the noun, and the GloVe Twitter vectors for
    # <user>/<url> (a reason those vectors were chosen) were never used.
    tokenizer_filters: str = string.punctuation.replace("<", "").replace(">", "") + "\t\n"
    early_stopping_patience: int = 2
    # Off by default, on evidence. The training split is 45% neutral and 16%
    # negative, and the linear baseline uses balanced weights, so weighting
    # these models looked obviously right - but measured, it trades a 2-3 point
    # gain on negative for a 6-10 point collapse on neutral, and costs both
    # models accuracy and macro F1. See MODEL_CARD.md. Set to "balanced" to
    # train weighted anyway; the training run records which was used.
    class_weight: str | None = None
    # Initialise the embedding from GloVe Twitter vectors instead of random
    # noise. Set to None to train embeddings from scratch.
    pretrained_embeddings: str | None = "glove-twitter-100"


@dataclass(frozen=True)
class RobertaConfig:
    """Fine-tuning settings for the Twitter-RoBERTa checkpoint."""

    base_model: str = "cardiffnlp/twitter-roberta-base-sentiment"
    max_len: int = 128
    batch_size: int = 8
    # The original run used 3 epochs; its trainer_state.json showed epoch 1 as
    # the best checkpoint (macro F1 0.7806) while validation loss climbed from
    # 0.5146 to 0.9767 by epoch 3. Retrains therefore stop at 2. In the
    # committed 2026-09-24 run epoch 2 was the better checkpoint on macro F1
    # (0.7864 against 0.7754); see reports/metrics/roberta_training_history.json.
    epochs: int = 2
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    early_stopping_patience: int = 1


LR_CONFIG = LRConfig()
SEQUENCE_CONFIG = SequenceConfig()
ROBERTA_CONFIG = RobertaConfig()
