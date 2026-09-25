"""Streamlit UI for the sentiment models.

Run it from the project root::

    streamlit run app/streamlit_app.py

Everything model-related is delegated to src.inference.predictor, so the UI, the
API and the tests share one preprocessing path and one definition of what a
probability vector means.

Models whose artifacts are missing or are Git LFS pointer stubs are kept out of
the selector entirely, with the reason shown in the sidebar. Offering a model
that is certain to raise on click is worse than not offering it.
"""

import sys
from pathlib import Path

# Make `streamlit run app/streamlit_app.py` work from the project root without
# installing the package. The previous version also called os.chdir() here,
# mutating process-global state as an import side effect.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.config import LABELS, METRICS_DIR
from src.inference.predictor import (
    LRPredictor,
    ModelUnavailableError,
    load_predictor,
    model_status,
)
from src.utils.io import load_json

TOP_TERMS = 10


@st.cache_resource(show_spinner="Loading model...")
def get_predictor(model_key: str):
    """Load a predictor once per session."""
    return load_predictor(model_key)


@st.cache_data(ttl=600)
def load_record(model_key: str) -> dict | None:
    """The model's metrics record, or None if it has none.

    Cached with a short ttl rather than forever: `make evaluate` rewrites the
    file, and a running UI should pick that up without a restart.
    """
    path = METRICS_DIR / f"{model_key}.json"
    if not path.exists():
        return None
    return load_json(path)


def render_confusion_matrix(model_key: str) -> None:
    """The model's test-split confusion matrix, from its metrics record."""
    record = load_record(model_key)
    metrics = (record or {}).get("metrics", {}).get("test")
    if not metrics or "confusion_matrix" not in metrics:
        st.info("No test-split metrics recorded for this model. Run `make evaluate`.")
        return

    matrix = metrics["confusion_matrix"]
    st.caption(
        f"Test split: accuracy {metrics['accuracy']:.4f}, macro F1 "
        f"{metrics['f1_macro']:.4f}. Rows are the true label, columns the "
        "predicted one."
    )
    st.dataframe(
        {
            "true \\ predicted": list(LABELS),
            **{label: [row[i] for row in matrix] for i, label in enumerate(LABELS)},
        },
        width="stretch",
        hide_index=True,
    )


def render_top_terms(predictor: LRPredictor, text: str, label: str) -> None:
    """The terms that pushed the linear model towards or away from its label."""
    terms = predictor.top_terms(text, label, k=TOP_TERMS)
    if not terms:
        st.info("None of the words in this text are in the model's vocabulary.")
        return

    st.subheader(f"Why {label}")
    st.caption(
        "Each bar is a term's tf-idf value times its coefficient for the "
        "predicted class: what it added to the score. Positive bars pushed "
        "towards the label, negative bars away from it."
    )
    chart = {
        "term": [t.term for t in terms],
        "contribution": [t.contribution for t in terms],
    }
    st.bar_chart(chart, x="term", y="contribution", horizontal=True, sort=False)
    st.dataframe(
        {
            "term": [t.term for t in terms],
            "weight": [t.weight for t in terms],
            "tf-idf": [t.tfidf for t in terms],
            "contribution": [t.contribution for t in terms],
        },
        width="stretch",
        hide_index=True,
        column_config={
            "weight": st.column_config.NumberColumn(format="%.3f"),
            "tf-idf": st.column_config.NumberColumn(format="%.3f"),
            "contribution": st.column_config.NumberColumn(format="%.3f"),
        },
    )


def render_sidebar(status: dict) -> str | None:
    """Model selector plus an explanation for anything unavailable."""
    usable = {key: info for key, info in status.items() if info["available"]}
    unusable = {key: info for key, info in status.items() if not info["available"]}

    choice = None
    if usable:
        choice = st.sidebar.selectbox(
            "Model",
            list(usable),
            format_func=lambda key: usable[key]["display_name"],
        )

    if unusable:
        with st.sidebar.expander(f"{len(unusable)} model(s) unavailable"):
            for info in unusable.values():
                st.write(f"**{info['display_name']}** - {info['reason']}")

    return choice


def main() -> None:
    st.set_page_config(page_title="Sentiment Analysis", layout="centered")
    st.title("Sentiment Analysis")
    st.caption("Three-class tweet sentiment: negative, neutral or positive.")

    status = model_status()
    choice = render_sidebar(status)

    if choice is None:
        st.error(
            "No model artifacts are usable. Run `make fetch-weights` "
            "to download the weights, or train one with "
            "`python -m src.training.train_lr`."
        )
        return

    text = st.text_area("Enter text to analyze:", height=140)

    predict = st.button("Predict", type="primary")

    # Rendered before the prediction, so it is there whether or not the user
    # has clicked yet; the record is a cached JSON read, not a model load.
    with st.expander("How this model does on the test split"):
        render_confusion_matrix(choice)

    if not predict:
        return

    if not text.strip():
        st.warning("Please enter some text.")
        return

    try:
        predictor = get_predictor(choice)
    except ModelUnavailableError as error:
        st.error(str(error))
        return

    try:
        prediction = predictor.predict(text)
    except Exception as error:  # noqa: BLE001 - the UI must not show a traceback
        st.error(f"Prediction failed: {error}")
        return

    st.subheader(f"Prediction: {prediction.label.upper()}")
    st.metric("Confidence", f"{prediction.confidence:.1%}")
    st.dataframe(
        {
            "Class": list(LABELS),
            "Probability": [prediction.probabilities[label] for label in LABELS],
        },
        width="stretch",
        hide_index=True,
    )

    # Only the linear model's decision is a sum of per-term weights that can
    # be shown exactly; the neural models get no explanation rather than a
    # misleading one.
    if isinstance(predictor, LRPredictor):
        render_top_terms(predictor, text, prediction.label)


main()
