import streamlit as st
import numpy as np
import joblib
from pathlib import Path
import sys
import os

# Ensure we can import from parent directory
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))
os.chdir(str(parent_dir))

from src.utils.preprocessing import preprocess_tweet
from src.models.bert_wrapper import BertWrapper


# -------------------------------------------------
# Streamlit Layout
# -------------------------------------------------
st.set_page_config(page_title="Sentiment Analysis", layout="centered")
st.title("Sentiment Analysis — LR, LSTM, GRU, BERT")

MODEL_CHOICES = ["LogisticRegression", "LSTM", "GRU", "BERT"]
model_choice = st.sidebar.selectbox("Select Model", MODEL_CHOICES)


# -------------------------------------------------
# Cached Loaders
# -------------------------------------------------

@st.cache_resource
def load_lr(path="models/lr/pipeline.joblib"):
    """Load Logistic Regression joblib pipeline."""
    return joblib.load(path)


@st.cache_resource
def load_lstm(path="models/lstm"):
    """Load tokenizer + LSTM Keras model."""
    from tensorflow.keras.models import load_model as _load_model

    tok_path = Path(path) / "tokenizer.joblib"
    model_path = Path(path) / "model_final.keras"

    if not tok_path.exists():
        raise FileNotFoundError("LSTM tokenizer.joblib missing in models/lstm/")
    if not model_path.exists():
        raise FileNotFoundError("LSTM model_final.keras missing in models/lstm/")

    tok = joblib.load(tok_path)
    mdl = _load_model(str(model_path))
    return tok, mdl


@st.cache_resource
def load_gru(path="models/gru"):
    """Load tokenizer + GRU Keras model."""
    from tensorflow.keras.models import load_model as _load_model

    tok_path = Path(path) / "tokenizer.joblib"
    model_path = Path(path) / "model_final.keras"

    if not tok_path.exists():
        raise FileNotFoundError("GRU tokenizer.joblib missing in models/gru/")
    if not model_path.exists():
        raise FileNotFoundError("GRU model_final.keras missing in models/gru/")

    tok = joblib.load(tok_path)
    mdl = _load_model(str(model_path))
    return tok, mdl


@st.cache_resource
def load_bert(path="models/bert"):
    """Load BERT model wrapper."""
    return BertWrapper(path)


# -------------------------------------------------
# Helper Functions
# -------------------------------------------------

def lstm_predict(tokenizer, model, text, max_len=80):
    """Prepare LSTM input and return softmax probabilities."""
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    seq = tokenizer.texts_to_sequences([text])
    seq = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")
    return model.predict(seq)[0]


def gru_predict(tokenizer, model, text, max_len=80):
    """Prepare GRU input and return softmax probabilities."""
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    seq = tokenizer.texts_to_sequences([text])
    seq = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")
    return model.predict(seq)[0]


# -------------------------------------------------
# Main UI
# -------------------------------------------------

st.write(f"Using Model: **{model_choice}**")
text = st.text_area("Enter text to analyze:", height=140)

if st.button("Predict"):
    if not text.strip():
        st.warning("Please enter some text.")
        st.stop()

    cleaned_text = preprocess_tweet(text)
    labels = ["negative", "neutral", "positive"]

    # -------------------------------
    # Logistic Regression
    # -------------------------------
    if model_choice == "LogisticRegression":
        try:
            model = load_lr()
        except Exception as e:
            st.error(f"LR model error: {e}")
            st.stop()

        probs = model.predict_proba([cleaned_text])[0]

    # -------------------------------
    # LSTM
    # -------------------------------
    elif model_choice == "LSTM":
        try:
            tokenizer, model = load_lstm()
            probs = lstm_predict(tokenizer, model, cleaned_text, max_len=80)
        except Exception as e:
            st.error(f"LSTM error: {e}")
            st.stop()

    # -------------------------------
    # GRU
    # -------------------------------
    elif model_choice == "GRU":
        try:
            tokenizer, model = load_gru()
            probs = gru_predict(tokenizer, model, cleaned_text, max_len=80)
        except Exception as e:
            st.error(f"GRU error: {e}")
            st.stop()

    # -------------------------------
    # BERT
    # -------------------------------
    else:
        try:
            bert = load_bert()
            probs = bert.predict_proba([cleaned_text])[0]
        except Exception as e:
            st.error(f"BERT error: {e}")
            st.stop()

    # -------------------------------
    # Output
    # -------------------------------
    pred_idx = int(np.argmax(probs))

    st.subheader(f"Prediction: **{labels[pred_idx].upper()}**")
    st.write(f"Confidence: `{float(probs[pred_idx]):.3f}`")

    st.table({
        "Class": labels,
        "Probability": [float(p) for p in probs]
    })
