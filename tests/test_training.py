"""Training-time behaviour that does not require running a training job."""

import numpy as np
import pytest

from src.config import LABELS, SEQUENCE_CONFIG
from src.training.train_sequence import compute_class_weights


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
