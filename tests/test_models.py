"""The Keras architectures.

Skipped entirely when no Keras backend is installed, so the suite still runs on
an inference-only machine.
"""

import warnings

import numpy as np
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


# --- masking -------------------------------------------------------------------


@pytest.mark.parametrize("name, build", BUILDERS)
def test_the_embedding_masks_padding(name, build):
    embedding = build(1_000).layers[0]
    assert embedding.mask_zero is True
    mask = embedding.compute_mask(
        np.array([[5, 6, 7] + [0] * (SEQUENCE_CONFIG.max_len - 3)])
    )
    assert mask is not None
    assert np.asarray(mask)[0].tolist() == [True] * 3 + [False] * (
        SEQUENCE_CONFIG.max_len - 3
    )


@pytest.mark.parametrize("name, build", BUILDERS)
def test_the_pretrained_branch_masks_too(name, build):
    matrix = np.zeros((1_000, SEQUENCE_CONFIG.embed_dim), dtype=np.float32)
    embedding = build(1_000, embedding_matrix=matrix).layers[0]
    assert embedding.mask_zero is True
    assert embedding.trainable is True


def test_both_architectures_share_one_embedding_builder():
    from src.models import embedding, gru_model, lstm_model

    assert lstm_model.build_embedding is embedding.build_embedding
    assert gru_model.build_embedding is embedding.build_embedding


@pytest.mark.parametrize("name, build", BUILDERS)
def test_padding_length_does_not_change_the_output(name, build):
    """The same tokens must score the same however much padding follows them.

    Without masking the pad vector runs through every step after the text
    ends, so 77 pads and 117 pads give different states. The pad vector is
    set large here so that difference is not lost in float noise.
    """
    short, long = build(100, max_len=80), build(100, max_len=120)
    weights = short.get_weights()
    weights[0][0] = 5.0
    short.set_weights(weights)
    long.set_weights(weights)

    tokens = [5, 6, 7]
    from_short = short.predict(np.array([tokens + [0] * 77]), verbose=0)
    from_long = long.predict(np.array([tokens + [0] * 117]), verbose=0)
    np.testing.assert_allclose(from_short, from_long, atol=1e-6)


@pytest.mark.parametrize("name, build", BUILDERS)
def test_a_text_that_tokenizes_to_nothing_still_gets_a_distribution(name, build):
    from src.utils.sequences import pad_sequences

    model = build(100)
    padded = pad_sequences([[]], SEQUENCE_CONFIG.max_len)
    probs = model.predict(padded, verbose=0)
    assert probs.shape == (1, NUM_LABELS)
    assert np.isfinite(probs).all()
    assert abs(probs.sum() - 1.0) < 1e-5
