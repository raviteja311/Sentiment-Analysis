"""The Streamlit UI, exercised headlessly with Streamlit's own AppTest harness."""

import pytest

from src.config import MODEL_DISPLAY_NAMES
from src.inference.predictor import available_models

pytest.importorskip("streamlit", reason="Streamlit is not installed")

from streamlit.testing.v1 import AppTest  # noqa: E402

APP = "../app/streamlit_app.py"
TIMEOUT = 60  # loading a model on first predict is slower than the 3s default


def run_app() -> AppTest:
    return AppTest.from_file(APP, default_timeout=TIMEOUT).run()


def test_app_starts_without_an_exception():
    assert not run_app().exception


def test_app_has_a_title():
    app = run_app()
    assert "Sentiment Analysis" in app.title[0].value


@pytest.mark.skipif(not available_models(), reason="no usable model artifacts")
def test_only_available_models_are_offered():
    # AppTest reports the options as the user sees them, after format_func.
    expected = [MODEL_DISPLAY_NAMES[key] for key in available_models()]
    app = run_app()
    assert list(app.sidebar.selectbox[0].options) == expected


@pytest.mark.skipif(not available_models(), reason="no usable model artifacts")
def test_empty_input_warns_instead_of_predicting():
    app = run_app()
    app.button[0].click().run()
    assert app.warning
    assert not app.exception


@pytest.mark.skipif("lr" not in available_models(), reason="LR artifact unavailable")
def test_predicting_shows_a_label_and_a_confidence():
    app = run_app()
    app.text_area[0].set_value("this is absolutely fantastic").run()
    app.button[0].click().run()

    assert not app.exception
    assert "Prediction:" in app.subheader[0].value
    assert app.metric[0].label == "Confidence"


@pytest.mark.skipif("lr" not in available_models(), reason="LR artifact unavailable")
def test_probability_table_is_rendered():
    app = run_app()
    app.text_area[0].set_value("this is awful").run()
    app.button[0].click().run()
    assert app.dataframe
