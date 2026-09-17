"""Entry point for training the Bi-LSTM.

The training loop itself is shared with the GRU model; see
src/training/train_sequence.py.
"""

import logging

from src.models.lstm_model import build_lstm
from src.training.train_sequence import train_sequence_model


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    train_sequence_model("lstm", build_lstm)


if __name__ == "__main__":
    main()
