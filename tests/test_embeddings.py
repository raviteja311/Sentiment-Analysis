"""Pretrained embedding matrices.

The 1 GB vector file is never touched here: load_vectors_for is stubbed, so
these tests run on any machine and in CI.
"""

import numpy as np
import pytest

from src import embeddings


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


def test_the_preprocessing_placeholders_are_covered(fake_vectors):
    # GloVe Twitter has <user> and <url>; our preprocessing emits them, which is
    # why these vectors were chosen over the Wikipedia-trained ones.
    _, coverage = embeddings.build_embedding_matrix({"<user>": 1}, vocab_size=2, dim=4)
    assert coverage == pytest.approx(1.0)


def test_matrix_is_float32(fake_vectors):
    matrix, _ = embeddings.build_embedding_matrix({"good": 1}, vocab_size=2, dim=4)
    assert matrix.dtype == np.float32


def test_builds_are_reproducible(fake_vectors):
    first, _ = embeddings.build_embedding_matrix({"zzz": 1}, vocab_size=2, dim=4, seed=7)
    second, _ = embeddings.build_embedding_matrix({"zzz": 1}, vocab_size=2, dim=4, seed=7)
    assert np.allclose(first, second)
