"""Unified inference interface for all four models.

The UI, the API and the tests all go through this module, so that text is
preprocessed the same way everywhere and a probability vector always means the
same thing: index ``i`` is the probability of ``config.LABELS[i]``.

Two failure modes are handled explicitly rather than left to surface as opaque
stack traces:

* **Git LFS pointer stubs.** A plain ``git clone`` without ``git lfs pull``
  leaves 133-byte text stubs where the weights should be. Loading one produces a
  deserialisation error that says nothing about LFS. :func:`is_lfs_pointer`
  detects them up front and the error names the fix.
* **Missing artifacts.** Same treatment - a model whose files are absent is
  reported as unavailable instead of crashing the caller at predict time.

Heavy backends (TensorFlow, PyTorch) are imported inside the loaders, so
importing this module stays cheap for callers that only want to know which
models exist.
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.config import (
    BERT_CONFIG,
    BERT_DIR,
    LABELS,
    LR_DIR,
    MODEL_DIRS,
    MODEL_DISPLAY_NAMES,
    MODEL_KEYS,
    NUM_LABELS,
    PROJECT_ROOT,
    REQUIRED_ARTIFACTS,
    RETRAIN_COMMANDS,
    SEQUENCE_CONFIG,
)
from src.utils.preprocessing import preprocess_tweet

# First bytes of a Git LFS pointer file, per the v1 pointer spec.
LFS_POINTER_PREFIX = b"version https://git-lfs"

LFS_HINT = "Run `git lfs install && git lfs pull` to download it"


class ModelUnavailableError(RuntimeError):
    """A model cannot be served: its artifacts are missing or are LFS stubs.

    Distinct from a genuine inference failure so that callers can answer with
    "this model is not loaded here" (HTTP 503) rather than "something broke"
    (HTTP 500).
    """


@dataclass(frozen=True)
class Prediction:
    """One classified text."""

    model: str
    label: str
    confidence: float
    probabilities: dict[str, float]


# ---------------------------------------------------------------------------
# Artifact checks
# ---------------------------------------------------------------------------


def is_lfs_pointer(path: str | Path) -> bool:
    """True if ``path`` is a Git LFS pointer stub rather than real content."""
    try:
        with open(path, "rb") as handle:
            head = handle.read(len(LFS_POINTER_PREFIX))
    except OSError:
        return False
    return head == LFS_POINTER_PREFIX


def _display_path(path: Path) -> str:
    """Path relative to the project root, with forward slashes."""
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def check_artifacts(model: str) -> str | None:
    """Return a human-readable reason the model cannot be loaded, or ``None``."""
    _require_known_model(model)
    retrain = RETRAIN_COMMANDS[model]

    for path in REQUIRED_ARTIFACTS[model]:
        if not path.exists():
            return (
                f"{_display_path(path)} is missing. {LFS_HINT}, "
                f"or retrain with `{retrain}`."
            )
        if is_lfs_pointer(path):
            return (
                f"{_display_path(path)} is a Git LFS pointer, not real model "
                f"weights. {LFS_HINT}, or retrain with `{retrain}`."
            )
    return None


def model_status() -> dict[str, dict]:
    """Availability of every known model, for health endpoints and the UI."""
    status = {}
    for model in MODEL_KEYS:
        reason = check_artifacts(model)
        status[model] = {
            "display_name": MODEL_DISPLAY_NAMES[model],
            "path": _display_path(MODEL_DIRS[model]),
            "available": reason is None,
            "reason": reason,
        }
    return status


def available_models() -> list[str]:
    """Models whose artifacts are present and are not LFS stubs."""
    return [model for model in MODEL_KEYS if check_artifacts(model) is None]


def _require_known_model(model: str) -> None:
    if model not in MODEL_KEYS:
        known = ", ".join(MODEL_KEYS)
        raise KeyError(f"Unknown model {model!r}. Known models: {known}.")


# ---------------------------------------------------------------------------
# Predictors
# ---------------------------------------------------------------------------


class Predictor:
    """Base class: preprocessing, batching and the probabilities -> label step."""

    key: str

    def __init__(self, key: str) -> None:
        self.key = key
        self.display_name = MODEL_DISPLAY_NAMES[key]
        reason = check_artifacts(key)
        if reason is not None:
            raise ModelUnavailableError(reason)

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """Class probabilities, shape ``(len(texts), NUM_LABELS)``."""
        if not texts:
            return np.zeros((0, NUM_LABELS), dtype=np.float32)
        cleaned = [preprocess_tweet(text) for text in texts]
        probs = np.asarray(self._predict_proba(cleaned), dtype=np.float32)
        if probs.shape != (len(texts), NUM_LABELS):
            raise RuntimeError(
                f"{self.key} returned probabilities of shape {probs.shape}, "
                f"expected {(len(texts), NUM_LABELS)}."
            )
        return probs

    def predict(self, text: str) -> Prediction:
        """Classify a single text."""
        probs = self.predict_proba([text])[0]
        index = int(np.argmax(probs))
        return Prediction(
            model=self.key,
            label=LABELS[index],
            confidence=float(probs[index]),
            probabilities={label: float(probs[i]) for i, label in enumerate(LABELS)},
        )

    def _predict_proba(self, cleaned: list[str]) -> np.ndarray:
        raise NotImplementedError


class LRPredictor(Predictor):
    """TF-IDF + LogisticRegression pipeline."""

    def __init__(self) -> None:
        super().__init__("lr")
        import joblib

        self._pipeline = joblib.load(LR_DIR / "pipeline.joblib")

    def _predict_proba(self, cleaned: list[str]) -> np.ndarray:
        probs = self._pipeline.predict_proba(cleaned)
        # predict_proba orders its columns by `classes_`, not by label index.
        # Today those coincide; relying on that was an unchecked assumption.
        return _order_by_label_index(probs, self._pipeline.classes_)


class SequencePredictor(Predictor):
    """Keras Bi-LSTM / Bi-GRU with its fitted tokenizer."""

    def __init__(self, key: str) -> None:
        super().__init__(key)
        import joblib
        from tensorflow.keras.models import load_model

        directory = MODEL_DIRS[key]
        self._tokenizer = joblib.load(directory / "tokenizer.joblib")
        self._model = load_model(str(directory / "model_final.keras"))

    def _predict_proba(self, cleaned: list[str]) -> np.ndarray:
        sequences = self._tokenizer.texts_to_sequences(cleaned)
        padded = _pad_sequences(sequences, SEQUENCE_CONFIG.max_len)
        return self._model.predict(padded, verbose=0)


class BertPredictor(Predictor):
    """Fine-tuned Twitter-RoBERTa checkpoint.

    The directory is named ``models/bert`` for historical reasons; the weights
    are RoBERTa (``model_type: roberta`` in its config).
    """

    def __init__(self, batch_size: int = 16, device: str | None = None) -> None:
        super().__init__("bert")
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self._batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._tokenizer = AutoTokenizer.from_pretrained(str(BERT_DIR), use_fast=True)
        self._model = AutoModelForSequenceClassification.from_pretrained(str(BERT_DIR))
        self._model.to(self.device)
        self._model.eval()

    def _predict_proba(self, cleaned: list[str]) -> np.ndarray:
        torch = self._torch
        batches = []
        for start in range(0, len(cleaned), self._batch_size):
            batch = cleaned[start : start + self._batch_size]
            encoded = self._tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=BERT_CONFIG.max_len,
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.no_grad():
                logits = self._model(**encoded).logits
            batches.append(torch.softmax(logits, dim=-1).cpu().numpy())
        return np.vstack(batches)


_PREDICTORS = {
    "lr": LRPredictor,
    "lstm": lambda: SequencePredictor("lstm"),
    "gru": lambda: SequencePredictor("gru"),
    "bert": BertPredictor,
}


@functools.cache
def load_predictor(model: str) -> Predictor:
    """Load a predictor, reusing the instance on subsequent calls.

    Raises :class:`ModelUnavailableError` if the artifacts are missing or are
    Git LFS pointer stubs.
    """
    _require_known_model(model)
    return _PREDICTORS[model]()


def clear_cache() -> None:
    """Drop cached predictors (used by tests and after retraining)."""
    load_predictor.cache_clear()


def predict(text: str, model: str) -> Prediction:
    """Convenience wrapper: load the model if needed and classify one text."""
    return load_predictor(model).predict(text)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _order_by_label_index(probs: np.ndarray, classes) -> np.ndarray:
    """Reorder scikit-learn ``predict_proba`` columns into label-index order."""
    ordered = np.zeros((probs.shape[0], NUM_LABELS), dtype=np.float32)
    for column, class_id in enumerate(classes):
        ordered[:, int(class_id)] = probs[:, column]
    return ordered


def _pad_sequences(sequences, max_len: int) -> np.ndarray:
    try:
        from keras.utils import pad_sequences
    except ImportError:  # pragma: no cover - Keras 2 layout
        from tensorflow.keras.preprocessing.sequence import pad_sequences

    return pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")
