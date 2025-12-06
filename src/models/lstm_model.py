from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout

def build_lstm(vocab_size, max_len, embed_dim=100):
  
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=max_len),
        Bidirectional(LSTM(128, return_sequences=True)),
        Dropout(0.35),
        Bidirectional(LSTM(64)),
        Dropout(0.25),
        Dense(128, activation="relu"),
        Dropout(0.2),
        Dense(3, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model
