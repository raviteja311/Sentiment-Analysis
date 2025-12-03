import streamlit as st
import joblib
import numpy as np
from src.utils.preprocessing import preprocess_tweet
from src.models.bert_wrapper import BertWrapper

st.set_page_config(page_title="Sentiment Analysis", layout="centered")
st.title("Sentiment Analysis — LR, BERT")

MODEL_CHOICES = ["LogisticRegression", "BERT"]
model_choice = st.sidebar.selectbox("Select Model", MODEL_CHOICES)

@st.cache_resource
def load_lr(path="models/lr/pipeline.joblib"):
    return joblib.load(path)

@st.cache_resource
def load_bert(path="models/bert"):
    return BertWrapper(path)

st.write(f"Using Model: **{model_choice}**")
text = st.text_area("Enter tweet text", height=120)
if st.button("Predict"):
    if not text.strip():
        st.warning("Please enter a valid text.")
        st.stop()
    t = preprocess_tweet(text)
    if model_choice == "LogisticRegression":
        try:
            pipe = load_lr()
        except Exception as e:
            st.error(f"LR model not found: {e}")
            st.stop()
        probs = pipe.predict_proba([t])[0]
        labels = ["negative","neutral","positive"]
    else:
        try:
            bert = load_bert()
        except Exception as e:
            st.error(f"BERT model not found: {e}")
            st.stop()
        probs = bert.predict_proba([t])[0]
        labels = ["negative","neutral","positive"]
    pred = int(np.argmax(probs))
    st.subheader(f"Prediction: **{labels[pred].upper()}**")
    st.write(f"Confidence: {float(probs[pred]):.3f}")
    st.table({"class": labels, "probability": [float(p) for p in probs]})
