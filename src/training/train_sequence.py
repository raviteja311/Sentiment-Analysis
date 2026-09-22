"""Shared training loop for the Bi-LSTM and Bi-GRU models.

train_lstm.py and train_gru.py were 95% identical: the same tokenizer fit, the
same padding, the same callbacks, the same evaluation, differing only in which
architecture they built and which directory they wrote to. Both now delegate
here and the divergence risk goes away.

The loop also persists its results. Previously every run computed metrics,
printed them and discarded them, which is why none of the numbers in the old
documentation could be checked against anything.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import asdict

import numpy as np

from src.config import (
    LABELS,
    METRICS_DIR,
    MODEL_DIRS,
    NUM_LABELS,
    SEED,
    SEQUENCE_CONFIG,
    SequenceConfig,
)
from src.data import describe, load_raw_dataset, prepare_split
from src.utils.io import save_json
from src.utils.metrics import build_report, compute_metrics

LOGGER = logging.getLogger(__name__)


def _tokenizer_class():
    """Return the Keras text Tokenizer class.

    Keras 3 removed the public ``keras.preprocessing.text`` alias, but the class
    itself survives as a legacy module - and it is the class the committed
    tokenizer.joblib files were pickled from, so using anything else would make
    new artifacts incompatible with the loaders. Prefer TensorFlow's public
    alias; fall back to the legacy path when only Keras is installed.
    """
    try:
        from tensorflow.keras.preprocessing.text import Tokenizer
    except ImportError:
        from keras.src.legacy.preprocessing.text import Tokenizer
    return Tokenizer


def fit_tokenizer(texts, config: SequenceConfig = SEQUENCE_CONFIG):
    """Fit a word tokenizer on the training texts."""
    tokenizer = _tokenizer_class()(num_words=config.max_vocab, oov_token=config.oov_token)
    tokenizer.fit_on_texts(texts)
    return tokenizer


def texts_to_padded(tokenizer, texts, max_len: int = SEQUENCE_CONFIG.max_len):
    """Encode texts as padded integer sequences of fixed length."""
    from keras.utils import pad_sequences

    return pad_sequences(
        tokenizer.texts_to_sequences(texts),
        maxlen=max_len,
        padding="post",
        truncating="post",
    )


def compute_class_weights(labels, strategy: str | None) -> dict[int, float] | None:
    """Per-class weights for an imbalanced training set.

    Returns None when ``strategy`` is None, which trains unweighted.
    """
    if not strategy:
        return None

    from sklearn.utils.class_weight import compute_class_weight

    classes = np.arange(NUM_LABELS)
    weights = compute_class_weight(strategy, classes=classes, y=np.asarray(labels))
    return {int(cls): float(weight) for cls, weight in zip(classes, weights, strict=True)}


def train_sequence_model(
    key: str,
    build_fn: Callable,
    config: SequenceConfig = SEQUENCE_CONFIG,
) -> dict:
    """Train one sequence model end to end and write its metrics record.

    Returns the record that was saved to ``reports/metrics/<key>.json``.
    """
    import joblib
    import keras
    from keras.callbacks import EarlyStopping, ModelCheckpoint
    from keras.utils import to_categorical

    keras.utils.set_random_seed(SEED)

    out_dir = MODEL_DIRS[key]
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_raw_dataset()
    splits = {
        name: prepare_split(dataset, name) for name in ("train", "validation", "test")
    }
    train_texts, train_labels = splits["train"]

    LOGGER.info("Fitting tokenizer (max_vocab=%d)...", config.max_vocab)
    tokenizer = fit_tokenizer(train_texts, config)
    joblib.dump(tokenizer, out_dir / "tokenizer.joblib")

    features = {
        name: texts_to_padded(tokenizer, texts, config.max_len)
        for name, (texts, _) in splits.items()
    }

    vocab_size = min(config.max_vocab, len(tokenizer.word_index) + 1)
    LOGGER.info(
        "Building %s (vocab=%d, embed=%d, max_len=%d)",
        key,
        vocab_size,
        config.embed_dim,
        config.max_len,
    )
    embedding_matrix, coverage = None, None
    if config.pretrained_embeddings:
        from src.embeddings import build_embedding_matrix

        embedding_matrix, coverage = build_embedding_matrix(
            tokenizer.word_index, vocab_size, dim=config.embed_dim
        )

    model = build_fn(
        vocab_size,
        max_len=config.max_len,
        embed_dim=config.embed_dim,
        embedding_matrix=embedding_matrix,
    )
    model.summary(print_fn=LOGGER.info)

    callbacks = [
        ModelCheckpoint(
            str(out_dir / "best.keras"), save_best_only=True, monitor="val_loss"
        ),
        # The old scripts paired 4 epochs with patience=3, so early stopping
        # could essentially never fire. Patience now comes from the config and
        # defaults to a value that can actually trigger.
        EarlyStopping(
            monitor="val_loss",
            patience=config.early_stopping_patience,
            restore_best_weights=True,
        ),
    ]

    class_weight = compute_class_weights(train_labels, config.class_weight)
    if class_weight:
        LOGGER.info(
            "Class weights (%s): %s",
            config.class_weight,
            {LABELS[i]: round(w, 3) for i, w in class_weight.items()},
        )

    LOGGER.info("Training...")
    model.fit(
        features["train"],
        to_categorical(train_labels, NUM_LABELS),
        validation_data=(
            features["validation"],
            to_categorical(splits["validation"][1], NUM_LABELS),
        ),
        epochs=config.epochs,
        batch_size=config.batch_size,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1,
    )

    final_path = out_dir / "model_final.keras"
    model.save(str(final_path))
    LOGGER.info("Saved final model: %s", final_path)

    metrics = {}
    for name in ("validation", "test"):
        predictions = np.argmax(
            model.predict(features[name], batch_size=config.batch_size, verbose=0),
            axis=1,
        )
        metrics[name] = compute_metrics(splits[name][1], predictions)
        LOGGER.info(
            "%s: accuracy=%.4f macro F1=%.4f",
            name,
            metrics[name]["accuracy"],
            metrics[name]["f1_macro"],
        )

    hyperparameters = asdict(config)
    if coverage is not None:
        hyperparameters["pretrained_embedding_coverage"] = round(coverage, 4)

    report = build_report(
        model=key,
        hyperparameters=hyperparameters,
        dataset=describe(dataset),
        metrics=metrics,
    )
    report_path = METRICS_DIR / f"{key}.json"
    save_json(report, report_path)
    LOGGER.info("Wrote metrics: %s", report_path)

    return report
