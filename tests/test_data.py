"""The held-out dev split carved from the training data."""

import numpy as np
import pytest

from src.config import DEV_FRACTION, LABELS
from src.data import class_distribution, dev_indices, with_dev_split


@pytest.fixture
def labels():
    # Roughly the training class mix: few negatives, many neutrals.
    rng = np.random.default_rng(0)
    return rng.choice(len(LABELS), size=4000, p=[0.16, 0.45, 0.39])


def test_dev_split_is_disjoint_and_covers_train(labels):
    train, dev = dev_indices(labels)
    assert not set(train) & set(dev)
    assert len(train) + len(dev) == len(labels)
    assert len(dev) == pytest.approx(DEV_FRACTION * len(labels), abs=1)


def test_dev_split_is_the_same_on_every_run(labels):
    np.testing.assert_array_equal(dev_indices(labels)[1], dev_indices(labels)[1])
    assert not np.array_equal(dev_indices(labels)[1], dev_indices(labels, seed=1)[1])


def test_dev_split_keeps_the_class_mix(labels):
    _, dev = dev_indices(labels)
    overall = np.bincount(labels) / len(labels)
    held_out = np.bincount(labels[dev], minlength=len(LABELS)) / len(dev)
    np.testing.assert_allclose(held_out, overall, atol=0.01)


def test_with_dev_split_leaves_validation_and_test_alone(labels):
    datasets = pytest.importorskip("datasets")
    dataset = datasets.DatasetDict(
        {
            "train": datasets.Dataset.from_dict(
                {"text": [str(i) for i in range(len(labels))], "label": labels}
            ),
            "validation": datasets.Dataset.from_dict({"text": ["v"], "label": [0]}),
            "test": datasets.Dataset.from_dict({"text": ["t"], "label": [1]}),
        }
    )

    carved = with_dev_split(dataset)

    assert len(carved["validation"]) == 1 and len(carved["test"]) == 1
    assert len(carved["train"]) + len(carved["dev"]) == len(labels)
    assert not set(carved["train"]["text"]) & set(carved["dev"]["text"])
    assert sum(class_distribution(carved["dev"]["label"]).values()) == len(carved["dev"])
