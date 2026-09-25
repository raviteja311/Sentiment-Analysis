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
    tables = [frame.value for frame in app.dataframe]
    assert any("Probability" in frame.columns for frame in tables)


# --- the confusion matrix ---------------------------------------------------


@pytest.mark.skipif("lr" not in available_models(), reason="LR artifact unavailable")
def test_the_confusion_matrix_is_shown_before_any_prediction():
    from src.config import LABELS

    app = run_app()
    assert not app.exception
    assert app.expander
    assert "test split" in app.expander[0].label
    frame = app.dataframe[0].value
    assert list(frame.columns)[1:] == list(LABELS)
    assert len(frame) == len(LABELS)
    # Counts, and there are 12,284 test tweets.
    assert int(frame[list(LABELS)].to_numpy().sum()) == 12284


# --- explaining the linear prediction ---------------------------------------


@pytest.mark.skipif("lr" not in available_models(), reason="LR artifact unavailable")
def test_the_linear_model_shows_the_terms_behind_its_label():
    app = run_app()
    app.text_area[0].set_value("this is absolutely awful").run()
    app.button[0].click().run()

    assert not app.exception
    assert any(sub.value.startswith("Why ") for sub in app.subheader)
    term_tables = [f.value for f in app.dataframe if "contribution" in f.value.columns]
    assert len(term_tables) == 1
    assert "awful" in list(term_tables[0]["term"])
