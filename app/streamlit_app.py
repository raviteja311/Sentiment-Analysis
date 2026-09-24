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

from src.config import LABELS
from src.inference.predictor import (
    ModelUnavailableError,
    load_predictor,
    model_status,
)


@st.cache_resource(show_spinner="Loading model...")
def get_predictor(model_key: str):
    """Load a predictor once per session."""
    return load_predictor(model_key)


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

    if not st.button("Predict", type="primary"):
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


main()
