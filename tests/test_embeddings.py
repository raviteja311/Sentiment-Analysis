"""Pretrained embedding matrices.

The 1 GB vector file is never touched here: load_vectors_for is stubbed, so
these tests run on any machine and in CI.
"""

import numpy as np
import pytest

from src import embeddings
from src.config import SEQUENCE_CONFIG


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


@pytest.mark.parametrize("model", ["lstm", "gru"])
def test_the_tokenizer_keeps_the_placeholder_brackets(model):
    """The vocabulary holds `<user>` and `<url>`, the GloVe Twitter tokens.

    preprocess_tweet emits those placeholders and GloVe Twitter has vectors by
    exactly those names. The Keras tokenizer's *default* filters include `<`
    and `>`, which silently turned them into the plain words "user" and "url" -
    so a mention was indistinguishable from the noun and the placeholder vectors
    were never used. SequenceConfig.tokenizer_filters now keeps the brackets;
    this pins the committed artifact to that behaviour.
    """
    import joblib

    from src.config import MODEL_DIRS

    path = MODEL_DIRS[model] / "tokenizer.joblib"
    if not path.is_file():
        pytest.skip("tokenizer not present")

    tokenizer = joblib.load(path)
    assert "<" not in tokenizer.filters and ">" not in tokenizer.filters
    assert tokenizer.filters == SEQUENCE_CONFIG.tokenizer_filters

    word_index = tokenizer.word_index
    assert "<user>" in word_index
    assert "<url>" in word_index
    # Within the embedding table, not beyond the cap where it would map to OOV.
    assert word_index["<user>"] < SEQUENCE_CONFIG.max_vocab


def test_configured_filters_keep_brackets_and_drop_the_rest():
    filters = SEQUENCE_CONFIG.tokenizer_filters
    assert "<" not in filters and ">" not in filters
    for char in "!#$%&()*+,-./:;=?@[]^_`{|}~":
        assert char in filters


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
