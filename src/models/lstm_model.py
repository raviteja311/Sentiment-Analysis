import keras
from keras.layers import LSTM, Bidirectional, Dense, Dropout, Embedding, Input
from keras.models import Sequential

from src.config import NUM_LABELS, SEQUENCE_CONFIG


def build_lstm(
    vocab_size,
    max_len=SEQUENCE_CONFIG.max_len,
    embed_dim=SEQUENCE_CONFIG.embed_dim,
) -> keras.Model:
    """Bi-LSTM classifier over a learned embedding.

    The input shape is declared with an explicit Input layer. It used to be
    passed as Embedding(input_length=max_len), which Keras 3 deprecated - it
    warned on every call and was ignored when building the graph.
    """
    model = Sequential(
        [
            Input(shape=(max_len,), dtype="int32"),
            Embedding(input_dim=vocab_size, output_dim=embed_dim),
            Bidirectional(LSTM(128, return_sequences=True)),
            Dropout(0.35),
            Bidirectional(LSTM(64)),
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
