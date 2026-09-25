"""Token sequences to fixed-length model input, shared by training and serving.

Training and the predictor used to pad separately, each through Keras's
``pad_sequences``. With masking on the embedding the two must agree on one
more thing: what to do with a text that tokenizes to nothing. This module
imports nothing beyond numpy, so the predictor can use it without pulling in
the training stack.
"""

from __future__ import annotations

import numpy as np

from src.config import SEQUENCE_CONFIG

# The Keras tokenizer assigns its oov_token index 1 (0 is reserved for padding).
OOV_INDEX = 1


def pad_sequences(
    sequences,
    max_len: int = SEQUENCE_CONFIG.max_len,
    empty_to_oov: bool = True,
) -> np.ndarray:
    """Post-pad and post-truncate integer sequences to ``max_len``.

    Matches ``keras.utils.pad_sequences(padding="post", truncating="post")``
    for non-empty input. An empty sequence, such as "!!!" once the tokenizer's
    filters have stripped it, becomes a single OOV token: with masking, an
    all-padding row is fully masked and a masked recurrent layer has no step
    to run on, so the model needs one real token to return a distribution
    from. ``empty_to_oov=False`` keeps the all-zero row, which is what the
    models trained before masking expect.
    """
    padded = np.zeros((len(sequences), max_len), dtype=np.int32)
    for row, sequence in enumerate(sequences):
        tokens = list(sequence)[:max_len]
        if not tokens and empty_to_oov:
            tokens = [OOV_INDEX]
        padded[row, : len(tokens)] = tokens
    return padded


def texts_to_padded(
    tokenizer,
    texts,
    max_len: int = SEQUENCE_CONFIG.max_len,
    empty_to_oov: bool = True,
) -> np.ndarray:
    """Encode texts with a fitted Keras tokenizer and pad them."""
    return pad_sequences(tokenizer.texts_to_sequences(texts), max_len, empty_to_oov)
