import numpy as np
from pathlib import Path
import joblib
from datasets import load_dataset
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from src.models.gru_model import build_gru
from src.utils.preprocessing import preprocess_tweet
from src.utils.metrics import compute_metrics

# ===========================
# FIXED HYPERPARAMETERS
# ===========================
OUT_DIR = "models/gru"
MAX_VOCAB = 10000
MAX_LEN = 80
EMBED_DIM = 100
BATCH = 64
EPOCHS = 4
SEED = 42
# ===========================

def preprocess_split(split):
    texts = [preprocess_tweet(t) for t in split["text"]]
    labels = np.array(split["label"])
    return texts, labels

def make_sequences(tokenizer, texts):
    seq = tokenizer.texts_to_sequences(texts)
    return pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")

def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    print("Loading dataset...")
    ds = load_dataset("cardiffnlp/tweet_eval", "sentiment")

    train_texts, train_labels = preprocess_split(ds["train"])
    val_texts, val_labels = preprocess_split(ds["validation"])
    test_texts, test_labels = preprocess_split(ds["test"])

    print(f"Fitting tokenizer (max_vocab={MAX_VOCAB})...")
    tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_texts)
    tok_path = Path(OUT_DIR) / "tokenizer.joblib"
    joblib.dump(tokenizer, tok_path)
    print("Saved tokenizer:", tok_path)

    print("Converting to sequences...")
    X_train = make_sequences(tokenizer, train_texts)
    X_val = make_sequences(tokenizer, val_texts)
    X_test = make_sequences(tokenizer, test_texts)

    y_train = tf.keras.utils.to_categorical(train_labels, 3)
    y_val = tf.keras.utils.to_categorical(val_labels, 3)

    vocab_size = min(MAX_VOCAB, len(tokenizer.word_index) + 1)
    print(f"Building GRU model (vocab_size={vocab_size}, embed_dim={EMBED_DIM})")

    model = build_gru(vocab_size=vocab_size, max_len=MAX_LEN, embed_dim=EMBED_DIM)
    model.summary()

    ckpt = ModelCheckpoint(str(Path(OUT_DIR) / "best.keras"), save_best_only=True, monitor="val_loss")
    es = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

    print("Training...")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH,
        callbacks=[ckpt, es],
        verbose=1
    )

    final_path = Path(OUT_DIR) / "model_final.keras"
    model.save(str(final_path))
    print("Saved final model:", final_path)

    print("Evaluating on test set...")
    preds = np.argmax(model.predict(X_test, batch_size=BATCH), axis=1)
    metrics = compute_metrics(test_labels, preds)
    print("Test metrics:", metrics)

if __name__ == "__main__":
    main()
