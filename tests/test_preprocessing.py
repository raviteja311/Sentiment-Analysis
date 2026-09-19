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


def test_keeps_hashtag_word_without_the_hash():
    assert preprocess_tweet("#Python is great") == "python is great"


def test_demojizes_emoji():
    result = preprocess_tweet("this is great 😀")
    assert "😀" not in result
    assert "grinning" in result


@pytest.mark.parametrize("value", [None, 42, 3.5, [], {}, object()])
def test_non_string_input_returns_empty_string(value):
    assert preprocess_tweet(value) == ""


def test_is_idempotent_on_already_clean_text():
    once = preprocess_tweet("the meeting is at noon")
    assert preprocess_tweet(once) == once


def test_combined_tweet():
    result = preprocess_tweet("@user LOVED #this http://a.b 😀")
    assert "<user>" in result
    assert "<url>" in result
    assert "this" in result
    assert result == result.strip()
