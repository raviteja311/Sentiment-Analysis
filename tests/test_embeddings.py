"""Pretrained embedding matrices.

The 1 GB vector file is never touched here: load_vectors_for is stubbed, so
these tests run on any machine and in CI.
"""

import numpy as np
import pytest

from src import embeddings
from src.config import LSTM_DIR


@pytest.fixture
def fake_vectors(monkeypatch):
    known = {
        "good": np.ones(4, dtype=np.float32),
        "bad": np.full(4, -1.0, dtype=np.float32),
        "<user>": np.full(4, 0.5, dtype=np.float32),
    }

    def fake_load(words, dim=4, **kwargs):
        return {w: v for w, v in known.items() if w in words}

    monkeypatch.setattr(embeddings, "load_vectors_for", fake_load)
    return known


def test_known_words_get_their_pretrained_vector(fake_vectors):
    word_index = {"good": 1, "bad": 2}
    matrix, _ = embeddings.build_embedding_matrix(word_index, vocab_size=3, dim=4)
    assert np.allclose(matrix[1], 1.0)
    assert np.allclose(matrix[2], -1.0)


def test_unknown_words_are_random_not_zero(fake_vectors):
    word_index = {"good": 1, "zzzunknown": 2}
    matrix, _ = embeddings.build_embedding_matrix(word_index, vocab_size=3, dim=4)
    # Zeros would make every unseen word identical and stop them separating.
    assert not np.allclose(matrix[2], 0.0)


def test_padding_row_is_zero(fake_vectors):
    matrix, _ = embeddings.build_embedding_matrix({"good": 1}, vocab_size=2, dim=4)
    assert np.allclose(matrix[0], 0.0)


def test_coverage_is_reported(fake_vectors):
    word_index = {"good": 1, "bad": 2, "zzzunknown": 3}
    _, coverage = embeddings.build_embedding_matrix(word_index, vocab_size=4, dim=4)
    assert coverage == pytest.approx(2 / 3)


def test_words_beyond_the_cap_are_ignored(fake_vectors):
    word_index = {"good": 1, "bad": 5}
    matrix, coverage = embeddings.build_embedding_matrix(word_index, vocab_size=2, dim=4)
    assert matrix.shape == (2, 4)
    assert coverage == pytest.approx(1.0)


@pytest.mark.skipif(
    not (LSTM_DIR / "tokenizer.joblib").is_file(), reason="tokenizer not present"
)
def test_the_tokenizer_strips_the_placeholder_brackets():
    """The vocabulary holds `user`, not `<user>`.

    preprocess_tweet emits `<user>` and `<url>`, and GloVe Twitter has tokens by
    those names - but the Keras tokenizer's default filters include `<` and `>`,
    so the brackets never reach the embedding lookup. An earlier version of this
    test hand-built a vocabulary containing `<user>` and therefore passed while
    the docstring claiming the match was wrong.
    """
    import joblib

    word_index = joblib.load(LSTM_DIR / "tokenizer.joblib").word_index
    assert "user" in word_index
    assert "<user>" not in word_index
    assert "url" in word_index
    assert "<url>" not in word_index


def test_matrix_is_float32(fake_vectors):
    matrix, _ = embeddings.build_embedding_matrix({"good": 1}, vocab_size=2, dim=4)
    assert matrix.dtype == np.float32


def test_builds_are_reproducible(fake_vectors):
    first, _ = embeddings.build_embedding_matrix({"zzz": 1}, vocab_size=2, dim=4, seed=7)
    second, _ = embeddings.build_embedding_matrix({"zzz": 1}, vocab_size=2, dim=4, seed=7)
    assert np.allclose(first, second)


@pytest.mark.parametrize("model", ["lstm", "gru"])
def test_tokenizers_do_not_depend_on_training_only_packages(model):
    """A committed tokenizer must load with the inference stack alone.

    `dill` arrives via `datasets`, which is a training dependency. A pickled
    `defaultdict(int)` picked up a `dill._dill` reference from dill's reducers,
    so the artifact only loaded on machines that had trained something. CI
    caught it; nothing local did, because dill was always installed here.
    """
    from src.config import MODEL_DIRS

    path = MODEL_DIRS[model] / "tokenizer.joblib"
    if not path.is_file():
        pytest.skip("tokenizer not present")
    assert b"dill._dill" not in path.read_bytes()
