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

import contextlib
import logging
import os
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.config import (
    LABELS,
    LR_DIR,
    MODEL_DIRS,
    MODEL_DISPLAY_NAMES,
    MODEL_KEYS,
    NUM_LABELS,
    PROJECT_ROOT,
    REQUIRED_ARTIFACTS,
    RETRAIN_COMMANDS,
    ROBERTA_CONFIG,
    ROBERTA_DIR,
    SEQUENCE_CONFIG,
    resolve_model,
)
from src.utils.preprocessing import artifact_spec, preprocess_tweet
from src.utils.sequences import texts_to_padded

LOGGER = logging.getLogger(__name__)

# First bytes of a Git LFS pointer file, per the v1 pointer spec.
LFS_POINTER_PREFIX = b"version https://git-lfs"

LFS_HINT = "Run `git lfs install && git lfs pull` to download it"
FETCH_HINT = "Run `make fetch-weights` to download it"

# Up to this many texts the Keras models run one predict_on_batch step; above
# it they go through model.predict, whose mini-batching pays off. See
# SequencePredictor.
FAST_PATH_MAX_TEXTS = 64


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


def inference_limit() -> int | None:
    """``MAX_CONCURRENT_INFERENCE`` as an int, or None for unlimited."""
    raw = os.environ.get("MAX_CONCURRENT_INFERENCE", "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        LOGGER.warning("MAX_CONCURRENT_INFERENCE=%r is not a number; ignored.", raw)
        return None
    return value if value > 0 else None


def build_inference_slots(limit: int | None) -> threading.BoundedSemaphore | None:
    return threading.BoundedSemaphore(limit) if limit else None


# Sync FastAPI endpoints run in a 40-thread pool, so forty simultaneous
# requests mean forty forward passes fighting for the same cores, every one of
# them slower than if they had queued. With a cap set, the surplus waits here
# instead of in the CPU scheduler. Unset means unlimited, which is the old
# behaviour.
_INFERENCE_SLOTS = build_inference_slots(inference_limit())


@contextlib.contextmanager
def _inference_slot():
    slots = _INFERENCE_SLOTS
    if slots is None:
        yield
        return
    with slots:
        yield


def configure_torch_threads(torch_module) -> int | None:
    """Apply ``TORCH_NUM_THREADS`` to torch, returning the count or None.

    Torch defaults to one intra-op thread per core, which on a shared node
    competes with everything else on the box and, combined with the request
    thread pool, oversubscribes it. Called once, at load, because
    ``set_num_threads`` is process-wide.
    """
    raw = os.environ.get("TORCH_NUM_THREADS", "").strip()
    if not raw:
        return None
    try:
        threads = int(raw)
    except ValueError:
        LOGGER.warning("TORCH_NUM_THREADS=%r is not a number; ignored.", raw)
        return None
    if threads <= 0:
        return None
    torch_module.set_num_threads(threads)
    return threads


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
    # Which artifact produced this. Without it a logged prediction cannot be
    # tied back to a model after the weights are retrained or swapped.
    version: str | None = None


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


def _recovery_hint(path: Path) -> str:
    """How to get this particular artifact back.

    The large weights are fetched from object storage; the small ones live in
    git, where an older clone may still have them as LFS pointer stubs.
    """
    from src.artifacts import is_remote

    if is_remote(path):
        return FETCH_HINT
    return LFS_HINT


def check_artifacts(model: str) -> str | None:
    """Return a human-readable reason the model cannot be loaded, or ``None``."""
    model = resolve_model(model)
    _require_known_model(model)
    retrain = RETRAIN_COMMANDS[model]

    for path in REQUIRED_ARTIFACTS[model]:
        if not path.exists():
            return (
                f"{_display_path(path)} is missing. {_recovery_hint(path)}, "
                f"or retrain with `{retrain}`."
            )
        if is_lfs_pointer(path):
            return (
                f"{_display_path(path)} is a Git LFS pointer, not real model "
                f"weights. {_recovery_hint(path)}, or retrain with `{retrain}`."
            )
    return None


def model_status() -> dict[str, dict]:
    """Availability of every known model, for health endpoints and the UI."""
    from src.inference.versioning import artifact_digests, model_version

    loaded = loaded_models()

    status = {}
    for model in MODEL_KEYS:
        reason = check_artifacts(model)
        on_disk = model_version(model) if reason is None else None
        in_memory = loaded[model].version if model in loaded else None

        status[model] = {
            "display_name": MODEL_DISPLAY_NAMES[model],
            "path": _display_path(MODEL_DIRS[model]),
            "available": reason is None,
            "reason": reason,
            # `version` is what predictions actually come from. A predictor is
            # cached once loaded, so replacing the weights or the calibration on
            # disk does not change what is being served - reporting the disk
            # version here would promise traceability the service cannot honour.
            "version": in_memory or on_disk,
            "loaded": model in loaded,
            "version_on_disk": on_disk,
            "stale": in_memory is not None and in_memory != on_disk,
            "artifacts": artifact_digests(model) if reason is None else {},
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
        from src.calibration import load_temperature

        self.key = key
        self.display_name = MODEL_DISPLAY_NAMES[key]
        reason = check_artifacts(key)
        if reason is not None:
            raise ModelUnavailableError(reason)

        # Raw softmax outputs are overconfident. If a temperature has been
        # fitted (`make calibrate`) it is applied here, so every consumer -
        # API, UI, evaluation - sees the same calibrated numbers. Temperature
        # scaling is monotonic, so this never changes a predicted label.
        self.temperature = load_temperature(key) or 1.0

        # Which text normalisation the artifact was trained with. Read from
        # the artifact, not from config: config says what the next training
        # run will use, and after a retrain the two differ until redeploy.
        try:
            self.preprocessing = artifact_spec(key)
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise ModelUnavailableError(
                f"{_display_path(MODEL_DIRS[key])}/preprocessing.json is unreadable: "
                f"{error}"
            ) from error

        from src.inference.versioning import model_version

        self.version = model_version(key)

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """Class probabilities, shape ``(len(texts), NUM_LABELS)``."""
        if not texts:
            return np.zeros((0, NUM_LABELS), dtype=np.float32)
        cleaned = [preprocess_tweet(text, self.preprocessing) for text in texts]
        with _inference_slot():
            probs = np.asarray(self._predict_proba(cleaned), dtype=np.float32)
        if probs.shape != (len(texts), NUM_LABELS):
            raise RuntimeError(
                f"{self.key} returned probabilities of shape {probs.shape}, "
                f"expected {(len(texts), NUM_LABELS)}."
            )

        if self.temperature != 1.0:
            from src.calibration import apply_temperature

            probs = apply_temperature(probs, self.temperature).astype(np.float32)
        return probs

    def predict_many(self, texts: Sequence[str]) -> list[Prediction]:
        """Classify several texts in one pass.

        One batched forward pass rather than one per text: for the transformer
        that is the difference between batching on the GPU and serialising.
        """
        return [self._to_prediction(row) for row in self.predict_proba(texts)]

    def predict(self, text: str) -> Prediction:
        """Classify a single text."""
        return self._to_prediction(self.predict_proba([text])[0])

    def _to_prediction(self, probs: np.ndarray) -> Prediction:
        index = int(np.argmax(probs))
        return Prediction(
            model=self.key,
            label=LABELS[index],
            confidence=float(probs[index]),
            probabilities={label: float(probs[i]) for i, label in enumerate(LABELS)},
            version=self.version,
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
        # A model trained with masking needs a text that tokenizes to nothing
        # replaced by one OOV token, or its row is fully masked. The models
        # trained before masking were never shown that and must keep getting
        # the all-zero row they were trained on, so the substitution follows
        # the artifact, not the current builder.
        self._masks_padding = bool(getattr(self._model.layers[0], "mask_zero", False))

    def _predict_proba(self, cleaned: list[str]) -> np.ndarray:
        padded = texts_to_padded(
            self._tokenizer,
            cleaned,
            SEQUENCE_CONFIG.max_len,
            empty_to_oov=self._masks_padding,
        )
        if len(padded) <= FAST_PATH_MAX_TEXTS:
            # model.predict builds a tf.data pipeline and an epoch loop on
            # every call, which for a handful of texts costs several times
            # the forward pass itself. predict_on_batch runs the same compiled
            # step function on the whole input at once, skipping that setup:
            # measured at 9 ms against 80 ms for one text on CPU. Calling the
            # model directly instead would be worse, not better - eager
            # execution runs the recurrent layers op by op and took 600 ms.
            return self._model.predict_on_batch(padded)
        return self._model.predict(padded, verbose=0)


class RobertaPredictor(Predictor):
    """Fine-tuned Twitter-RoBERTa checkpoint."""

    def __init__(self, batch_size: int = 16, device: str | None = None) -> None:
        super().__init__("roberta")
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self._batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        configure_torch_threads(torch)

        self._tokenizer = AutoTokenizer.from_pretrained(str(ROBERTA_DIR), use_fast=True)
        self._model = AutoModelForSequenceClassification.from_pretrained(str(ROBERTA_DIR))
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
                max_length=ROBERTA_CONFIG.max_len,
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            # inference_mode goes further than no_grad: it also skips the
            # version counters and view tracking autograd keeps on tensors.
            with torch.inference_mode():
                logits = self._model(**encoded).logits
            batches.append(torch.softmax(logits, dim=-1).cpu().numpy())
        return np.vstack(batches)


_PREDICTORS = {
    "lr": LRPredictor,
    "lstm": lambda: SequencePredictor("lstm"),
    "gru": lambda: SequencePredictor("gru"),
    "roberta": RobertaPredictor,
}


_LOADED: dict[str, Predictor] = {}

# functools.cache only locks its own bookkeeping, not the call, so four
# simultaneous first requests build four models - a ~2 GB spike for the
# transformer. One lock per key serialises the first load and lets subsequent
# readers past.
_LOAD_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def _lock_for(model: str) -> threading.Lock:
    with _LOCKS_GUARD:
        return _LOAD_LOCKS.setdefault(model, threading.Lock())


def load_predictor(model: str) -> Predictor:
    """Load a predictor by key, resolving deprecated aliases first.

    Raises :class:`ModelUnavailableError` if the artifacts are missing or are
    Git LFS pointer stubs.
    """
    model = resolve_model(model)
    cached = _LOADED.get(model)
    if cached is not None:
        return cached

    # Validate before taking a lock: _lock_for creates an entry per name, so an
    # unknown name used to leave a lock behind in _LOAD_LOCKS for every call.
    _require_known_model(model)

    with _lock_for(model):
        cached = _LOADED.get(model)
        if cached is not None:
            return cached
        predictor = _PREDICTORS[model]()
        _LOADED[model] = predictor
        return predictor


def loaded_models() -> dict[str, Predictor]:
    """Predictors currently held in memory, keyed by canonical name."""
    return dict(_LOADED)


def clear_cache() -> None:
    """Drop cached predictors (used by tests and after retraining)."""
    _LOADED.clear()


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
