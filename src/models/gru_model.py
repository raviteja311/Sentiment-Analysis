import keras
from keras.layers import GRU, Bidirectional, Dense, Dropout, Embedding, Input
from keras.models import Sequential

from src.config import NUM_LABELS, SEQUENCE_CONFIG


def _embedding(vocab_size, embed_dim, matrix):
    """Embedding layer, optionally initialised from pretrained vectors.

    Pretrained weights stay trainable: the vectors are a starting point,
    and freezing them would stop the model adapting to this dataset.
    """
    if matrix is None:
        return Embedding(input_dim=vocab_size, output_dim=embed_dim)
    return Embedding(
        input_dim=vocab_size,
        output_dim=embed_dim,
        embeddings_initializer=keras.initializers.Constant(matrix),
        trainable=True,
    )


def build_gru(
    vocab_size,
    max_len=SEQUENCE_CONFIG.max_len,
    embed_dim=SEQUENCE_CONFIG.embed_dim,
    embedding_matrix=None,
) -> keras.Model:
    """Bi-GRU classifier over a learned embedding.

    Mirrors build_lstm; see that docstring for why the input shape is declared
    with an explicit Input layer rather than Embedding(input_length=...).
    """
    model = Sequential(
        [
            Input(shape=(max_len,), dtype="int32"),
            _embedding(vocab_size, embed_dim, embedding_matrix),
            Bidirectional(GRU(128, return_sequences=True)),
            Dropout(0.35),
            Bidirectional(GRU(64)),
            Dropout(0.25),
            Dense(128, activation="relu"),
            Dropout(0.2),
            Dense(NUM_LABELS, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
