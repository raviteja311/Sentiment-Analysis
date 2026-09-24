"""Pretrained word vectors for the recurrent models.

The Bi-LSTM and Bi-GRU learn their embeddings from 45,615 tweets, which is not
enough text to discover that "brilliant" and "superb" mean similar things. The
transformer does not have that problem because it arrives knowing; giving the
recurrent models the same head start is the one remaining lever on their weakest
number, positive-class recall.

GloVe Twitter vectors are used rather than the more common Wikipedia ones
because they were trained on tweets: the vocabulary covers slang, hashtag words
and the register of the dataset, which Wikipedia-trained vectors do not.

One caveat worth knowing. GloVe Twitter contains the placeholder tokens
``<user>`` and ``<url>``, and :func:`src.utils.preprocessing.preprocess_tweet`
emits exactly those - but the Keras tokenizer's default ``filters`` strip ``<``
and ``>``, so the fitted vocabulary holds ``user`` and ``url`` instead. Those
plain words do exist in GloVe, so they are covered, but the vectors carry the
everyday meanings rather than the Twitter placeholder ones. Mapping them back
would need a retrain to take effect and has not been measured.

The vectors are ~1 GB of text, downloaded on demand and cached by
``huggingface_hub``; only the rows matching the fitted tokenizer are kept, so
nothing large is held in memory or written into the repository.
"""

from __future__ import annotations

import logging

import numpy as np

LOGGER = logging.getLogger(__name__)

# Plain-text GloVe, deliberately not the gensim-pickled copies: reading it needs
# nothing beyond the standard library, so no extra dependency reaches the
# pinned environment.
GLOVE_REPO = "jkrukowski/glove-twitter-100"
GLOVE_FILE = "glove-twitter-100.txt"
GLOVE_DIM = 100


def load_vectors_for(
    words: set[str],
    dim: int = GLOVE_DIM,
    repo_id: str = GLOVE_REPO,
    filename: str = GLOVE_FILE,
) -> dict[str, np.ndarray]:
    """Read vectors for ``words`` only, streaming the file line by line."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=repo_id, filename=filename)
    LOGGER.info("Reading %s for %d words...", filename, len(words))

    found: dict[str, np.ndarray] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            token, _, rest = line.partition(" ")
            if token in words:
                values = np.fromstring(rest, sep=" ", dtype=np.float32)
                if values.size == dim:
                    found[token] = values
                if len(found) == len(words):
                    break
    return found


def build_embedding_matrix(
    word_index: dict[str, int],
    vocab_size: int,
    dim: int = GLOVE_DIM,
    seed: int = 42,
    repo_id: str = GLOVE_REPO,
    filename: str = GLOVE_FILE,
) -> tuple[np.ndarray, float]:
    """Embedding matrix aligned to a Keras tokenizer's ``word_index``.

    Returns the matrix and the fraction of the vocabulary that was covered.
    Words with no pretrained vector keep a small random vector rather than
    zeros, so they stay trainable and distinguishable from padding.
    """
    wanted = {word for word, index in word_index.items() if index < vocab_size}
    vectors = load_vectors_for(wanted, dim=dim, repo_id=repo_id, filename=filename)

    rng = np.random.default_rng(seed)
    matrix = rng.normal(0.0, 0.1, size=(vocab_size, dim)).astype(np.float32)
    matrix[0] = 0.0  # padding index

    for word, index in word_index.items():
        if index < vocab_size and word in vectors:
            matrix[index] = vectors[word]

    coverage = len(vectors) / max(len(wanted), 1)
    LOGGER.info(
        "Pretrained vectors cover %.1f%% of the vocabulary (%d of %d words)",
        coverage * 100,
        len(vectors),
        len(wanted),
    )
    return matrix, coverage
