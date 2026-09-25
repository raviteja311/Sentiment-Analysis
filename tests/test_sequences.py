"""Padding, shared by training and serving."""

import subprocess
import sys

import numpy as np
import pytest

from src.config import PROJECT_ROOT
from src.utils.sequences import OOV_INDEX, pad_sequences, texts_to_padded


def test_pads_after_the_tokens_and_truncates_the_tail():
    padded = pad_sequences([[1, 2, 3], [4, 5, 6, 7, 8]], max_len=4)
    assert padded.tolist() == [[1, 2, 3, 0], [4, 5, 6, 7]]
    assert padded.dtype == np.int32


def test_an_empty_sequence_becomes_one_oov_token():
    # A fully padded row is fully masked, and a masked recurrent layer has no
    # step to run on.
    assert pad_sequences([[]], max_len=3).tolist() == [[OOV_INDEX, 0, 0]]


def test_the_substitution_can_be_switched_off_for_models_trained_without_masking():
    assert pad_sequences([[]], max_len=3, empty_to_oov=False).tolist() == [[0, 0, 0]]


def test_no_input_gives_an_empty_matrix():
    assert pad_sequences([], max_len=5).shape == (0, 5)


def test_the_default_length_comes_from_the_config():
    from src.config import SEQUENCE_CONFIG

    assert pad_sequences([[1]]).shape == (1, SEQUENCE_CONFIG.max_len)


def test_matches_keras_for_non_empty_input():
    keras = pytest.importorskip("keras")
    rows = [[1, 2], [3, 4, 5, 6, 7, 8, 9], [2]]
    expected = keras.utils.pad_sequences(
        rows, maxlen=5, padding="post", truncating="post"
    )
    np.testing.assert_array_equal(pad_sequences(rows, max_len=5), expected)


def test_texts_to_padded_goes_through_the_tokenizer():
    class Tokenizer:
        def texts_to_sequences(self, texts):
            return [[len(text)] if text else [] for text in texts]

    padded = texts_to_padded(Tokenizer(), ["abc", ""], max_len=2)
    assert padded.tolist() == [[3, 0], [OOV_INDEX, 0]]


def test_the_module_needs_no_training_stack():
    # The predictor imports it, and the predictor must load without Keras.
    code = (
        "import sys; import src.utils.sequences; "
        "assert 'keras' not in sys.modules and 'tensorflow' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=PROJECT_ROOT)
