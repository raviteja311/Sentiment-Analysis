"""Integrated Gradients for the transformer."""

import pytest

from src.config import ROBERTA_DIR
from src.inference.attributions import (
    TokenAttribution,
    integrated_gradients,
    merge_into_words,
)
from tests.conftest import requires_model

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def tokenizer():
    # The committed tokenizer files; no weights or network needed.
    return transformers.AutoTokenizer.from_pretrained(str(ROBERTA_DIR), use_fast=True)


@pytest.fixture(scope="module")
def tiny_model(tokenizer):
    # A randomly initialised RoBERTa small enough to run in milliseconds. The
    # algorithm's guarantees hold for any differentiable model, trained or not.
    torch.manual_seed(0)
    config = transformers.RobertaConfig(
        vocab_size=len(tokenizer),
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=136,
        num_labels=3,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = transformers.RobertaForSequenceClassification(config)
    model.eval()
    return model


def test_attributions_are_complete(tiny_model, tokenizer):
    # Completeness: the scores sum to F(input) - F(baseline).
    result = integrated_gradients(tiny_model, tokenizer, "not good at all", steps=200)
    assert abs(result.convergence_delta) < 1e-3


def test_more_steps_converge_further(tiny_model, tokenizer):
    text = "what a wonderful, terrible day"
    coarse = integrated_gradients(tiny_model, tokenizer, text, steps=2)
    fine = integrated_gradients(tiny_model, tokenizer, text, steps=200)
    assert abs(fine.convergence_delta) <= abs(coarse.convergence_delta)


def test_one_score_per_content_token(tiny_model, tokenizer):
    text = "i love it"
    result = integrated_gradients(tiny_model, tokenizer, text, steps=4)
    content = tokenizer.tokenize(text)
    assert [token.token for token in result.tokens] == content
    assert [word.token for word in result.words] == ["i", "love", "it"]


def test_target_defaults_to_the_predicted_class(tiny_model, tokenizer):
    from src.config import LABELS

    text = "meh"
    with torch.no_grad():
        logits = tiny_model(**tokenizer(text, return_tensors="pt")).logits
    result = integrated_gradients(tiny_model, tokenizer, text, steps=4)
    assert result.label == LABELS[int(logits.argmax())]


def test_explaining_does_not_touch_the_weights(tiny_model, tokenizer):
    integrated_gradients(tiny_model, tokenizer, "fine", steps=4)
    assert all(parameter.grad is None for parameter in tiny_model.parameters())


def test_sub_words_are_summed_into_words():
    tokens = [
        TokenAttribution("Ġun", 0.25),
        TokenAttribution("believ", 0.5),
        TokenAttribution("able", 0.25),
        TokenAttribution("Ġday", -1.0),
    ]
    words = merge_into_words(tokens)
    assert words == [TokenAttribution("unbelievable", 1.0), TokenAttribution("day", -1.0)]


def test_first_word_without_a_space_marker_is_kept():
    words = merge_into_words([TokenAttribution("hi", 0.1), TokenAttribution("Ġx", 0.2)])
    assert [w.token for w in words] == ["hi", "x"]


@requires_model("roberta")
def test_the_served_model_blames_the_right_words():
    from src.inference.attributions import explain_roberta
    from src.inference.predictor import load_predictor

    predictor = load_predictor("roberta")
    result = explain_roberta(predictor, "I hate this phone, it is awful")

    assert result.label == "negative"
    assert abs(result.convergence_delta) < 0.1
    top = max(result.words, key=lambda word: word.score).token
    assert top in {"hate", "awful"}
