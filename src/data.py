"""Shared dataset loading for the training scripts.

Every training script used to carry its own copy of the load-and-preprocess
logic, and none of them recorded what they had actually loaded. The prose
documentation consequently described a training split of ~6,000 examples when
the real one is 45,615 - a 7.6x understatement that nothing in the code would
have contradicted.

So the loader here logs the real split sizes on every run, and exposes
:func:`describe` so a training script can record them alongside its metrics.
"""

from __future__ import annotations

import logging
from collections import Counter

import numpy as np

from src.config import DATASET_CONFIG, DATASET_NAME, LABELS, SPLITS
from src.utils.preprocessing import preprocess_tweet

LOGGER = logging.getLogger(__name__)


def load_raw_dataset(name: str = DATASET_NAME, config: str = DATASET_CONFIG):
    """Load the dataset and log the size of each split.

    ``datasets`` is imported lazily: it is a training-only dependency, and this
    module is also imported by code paths that never touch the dataset.
    """
    from datasets import load_dataset

    LOGGER.info("Loading dataset %s (%s)...", name, config)
    dataset = load_dataset(name, config)

    for split, size in split_sizes(dataset).items():
        LOGGER.info("  %-10s %7d examples", split, size)

    return dataset


def prepare_split(dataset, split: str, spec: str) -> tuple[list[str], np.ndarray]:
    """Return texts normalised with preprocessing ``spec``, and integer labels.

    The spec is required rather than defaulted so that no training run can
    silently pick up the wrong one; pass ``config.TRAIN_PREPROCESSING[key]``.
    """
    texts = [preprocess_tweet(text, spec) for text in dataset[split]["text"]]
    labels = np.asarray(dataset[split]["label"], dtype=np.int64)
    return texts, labels


def load_splits(
    spec: str,
    splits: tuple[str, ...] = SPLITS,
    name: str = DATASET_NAME,
    config: str = DATASET_CONFIG,
) -> dict[str, tuple[list[str], np.ndarray]]:
    """Load the dataset and return ``{split: (texts, labels)}``."""
    dataset = load_raw_dataset(name=name, config=config)
    return {split: prepare_split(dataset, split, spec) for split in splits}


def split_sizes(dataset) -> dict[str, int]:
    """Number of examples in each split, in the dataset's own order."""
    return {split: len(dataset[split]) for split in dataset}


def class_distribution(labels) -> dict[str, int]:
    """Count examples per class, keyed by label name rather than index."""
    counts = Counter(int(label) for label in labels)
    return {name: counts.get(index, 0) for index, name in enumerate(LABELS)}


def describe(dataset) -> dict:
    """Summarise the loaded dataset for the metrics report.

    The result is JSON-serialisable and is meant to be stored next to a model's
    scores, so that a number in the README can always be traced back to the data
    it was measured on.
    """
    return {
        "name": DATASET_NAME,
        "config": DATASET_CONFIG,
        "split_sizes": split_sizes(dataset),
        "class_distribution": {
            split: class_distribution(dataset[split]["label"]) for split in dataset
        },
    }
