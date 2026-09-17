"""Entry point for training the Bi-GRU.

The training loop itself is shared with the LSTM model; see
src/training/train_sequence.py.
"""

import logging

from src.models.gru_model import build_gru
from src.training.train_sequence import train_sequence_model


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    train_sequence_model("gru", build_gru)


if __name__ == "__main__":
    main()
