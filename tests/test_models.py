"""The Keras architectures.

Skipped entirely when no Keras backend is installed, so the suite still runs on
an inference-only machine.
"""

import warnings

import pytest

from src.config import NUM_LABELS, SEQUENCE_CONFIG

pytest.importorskip("keras", reason="Keras (with a backend) is not installed")

from src.models.gru_model import build_gru  # noqa: E402
from src.models.lstm_model import build_lstm  # noqa: E402

BUILDERS = [("lstm", build_lstm), ("gru", build_gru)]


@pytest.mark.parametrize("name, build", BUILDERS)
def test_input_shape_comes_from_the_config(name, build):
    model = build(1_000)
    assert model.input_shape == (None, SEQUENCE_CONFIG.max_len)


@pytest.mark.parametrize("name, build", BUILDERS)
def test_output_is_one_probability_per_label(name, build):
    model = build(1_000)
    assert model.output_shape == (None, NUM_LABELS)


@pytest.mark.parametrize("name, build", BUILDERS)
def test_builds_without_deprecation_warnings(name, build):
    # Embedding(input_length=...) used to warn on every call under Keras 3.
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        warnings.simplefilter("error", UserWarning)
        build(1_000)


@pytest.mark.parametrize("name, build", BUILDERS)
def test_respects_an_explicit_max_len(name, build):
    model = build(1_000, max_len=32)
    assert model.input_shape == (None, 32)


@pytest.mark.parametrize("name, build", BUILDERS)
def test_embedding_dimension_is_configurable(name, build):
    model = build(500, embed_dim=16)
    embedding = model.layers[0]
    assert embedding.input_dim == 500
    assert embedding.output_dim == 16


@pytest.mark.parametrize("name, build", BUILDERS)
def test_model_is_compiled_for_training(name, build):
    model = build(1_000)
    assert model.optimizer is not None
    assert model.loss == "categorical_crossentropy"


@pytest.mark.parametrize("name, build", BUILDERS)
def test_final_layer_is_a_softmax(name, build):
    model = build(1_000)
    assert model.layers[-1].activation.__name__ == "softmax"
