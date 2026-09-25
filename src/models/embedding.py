"""The embedding layer shared by the Bi-LSTM and Bi-GRU.

Both architectures used to carry an identical private copy of this; one place
means one place to get masking right.
"""

import keras
from keras.layers import Embedding


def build_embedding(vocab_size, embed_dim, matrix=None) -> Embedding:
    """Embedding layer, optionally initialised from pretrained vectors.

    ``mask_zero=True`` so the recurrent layers skip padding. Without it the
    pad vector was fed through every recurrent step after the text ended, so
    a representation depended on how much padding followed it: the same
    tokens scored differently under ``max_len=80`` and ``max_len=120``. Index 0
    is what the Keras tokenizer reserves for padding, so nothing real is
    masked.

    Pretrained weights stay trainable: the vectors are a starting point, and
    freezing them would stop the model adapting to this dataset.
    """
    kwargs = {"input_dim": vocab_size, "output_dim": embed_dim, "mask_zero": True}
    if matrix is not None:
        kwargs["embeddings_initializer"] = keras.initializers.Constant(matrix)
        kwargs["trainable"] = True
    return Embedding(**kwargs)
