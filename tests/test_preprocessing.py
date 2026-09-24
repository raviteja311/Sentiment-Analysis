"""Preprocessing is the one step every model shares, so it gets the most tests."""

import pytest

from src.utils.preprocessing import preprocess_tweet


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("HELLO World", "hello world"),
        ("  padded  ", "padded"),
        ("multiple    spaces", "multiple spaces"),
        ("line\nbreak", "line break"),
        ("", ""),
    ],
)
def test_normalises_whitespace_and_case(raw, expected):
    assert preprocess_tweet(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "check http://example.com now",
        "check https://example.com/path?q=1 now",
        "see www.example.com today",
        "SEE WWW.EXAMPLE.COM TODAY",
    ],
)
def test_replaces_urls_with_placeholder(raw):
    result = preprocess_tweet(raw)
    assert "<url>" in result
    assert "example.com" not in result


def test_replaces_mentions_with_placeholder():
    result = preprocess_tweet("thanks @someone for the help")
    assert "<user>" in result
    assert "@someone" not in result


@pytest.mark.parametrize(
    "raw",
    [
        "mail jane@gmail.com please",
        "contact bob.smith+tag@mail.co.uk now",
    ],
)
def test_an_email_is_not_a_mention(raw):
    # `@\w+` alone turned "jane@gmail.com" into "jane <user> .com": it invented
    # a mention and left the rest as debris.
    result = preprocess_tweet(raw)
    assert "<email>" in result
    assert "<user>" not in result
    assert "gmail" not in result


def test_a_mention_after_a_word_boundary_still_works():
    assert "<user>" in preprocess_tweet("hey @bob!")


def test_keeps_hashtag_word_without_the_hash():
    assert preprocess_tweet("#Python is great") == "python is great"


def test_emoji_become_ordinary_words():
    # ":smiling_face_with_heart-eyes:" used to survive tokenization as one glued
    # out-of-vocabulary term. Split into words, each has a pretrained vector -
    # which matters because 6.6% of the test split contains emoji and the
    # training split contains none at all.
    result = preprocess_tweet("love it \U0001f60d")
    assert "smiling" in result and "face" in result
    assert "_" not in result
    assert ":" not in result


def test_a_single_word_emoji_stays_one_word():
    assert "fire" in preprocess_tweet("this is \U0001f525")


@pytest.mark.parametrize("value", [None, 42, 3.5, [], {}, object()])
def test_non_string_input_returns_empty_string(value):
    assert preprocess_tweet(value) == ""


def test_is_idempotent_on_already_clean_text():
    once = preprocess_tweet("the meeting is at noon")
    assert preprocess_tweet(once) == once


def test_placeholders_survive_a_second_pass():
    # The predictor preprocesses whatever it is given, so a placeholder must not
    # be mangled if the text is run through twice.
    once = preprocess_tweet("@bob see http://x.co")
    assert preprocess_tweet(once) == once


def test_combined_tweet():
    result = preprocess_tweet("@user LOVED #this http://a.b \U0001f60d")
    assert "<user>" in result
    assert "<url>" in result
    assert "this" in result
    assert result == result.strip()
