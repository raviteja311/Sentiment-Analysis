"""Preprocessing is the one step every model shares, so it gets the most tests."""

import json

import pytest

from src.utils import preprocessing
from src.utils.preprocessing import SPECS, preprocess_tweet


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


# --- specs -----------------------------------------------------------------
#
# Preprocessing is versioned. The tests above exercise the default, glove-v1,
# and must keep passing unchanged: the shipped artifacts were trained with it.


def test_the_default_spec_is_glove_v1():
    raw = "@bob LOVED #ThisThing http://a.b can't"
    assert preprocess_tweet(raw) == preprocess_tweet(raw, "glove-v1")


def test_the_registry_lists_every_spec():
    assert set(SPECS) == {"glove-v1", "glove-v2", "cardiff-v1"}


def test_an_unknown_spec_is_rejected_by_name():
    with pytest.raises(ValueError, match="glove-v9") as excinfo:
        preprocess_tweet("hello", "glove-v9")
    # The message lists what would have worked.
    assert "glove-v2" in str(excinfo.value)


def test_an_unknown_spec_is_rejected_even_for_non_string_input():
    with pytest.raises(ValueError):
        preprocess_tweet(None, "nope")


# Golden values captured from the original implementation. If one of these
# changes, glove-v1 has changed, and every artifact on disk is now being shown
# text it was not trained on.
GLOVE_V1_GOLDEN = [
    (
        "@user LOVED #this http://a.b \U0001f60d",
        "<user> loved this <url> smiling face with heart-eyes",
    ),
    ("I can't stand this &amp; I won't lie", "i can't stand this &amp; i won't lie"),
    ("&lt;3 you #NotHappy don\u2019t", "&lt;3 you nothappy don\u2019t"),
    ("mail jane@gmail.com please, see www.example.com", "mail <email> please, see <url>"),
    ("  Multiple   spaces\nand CAPS  ", "multiple spaces and caps"),
    ("this is \U0001f525 #BREAKING #python", "this is fire breaking python"),
]


@pytest.mark.parametrize("raw, expected", GLOVE_V1_GOLDEN)
def test_glove_v1_is_frozen(raw, expected):
    assert preprocess_tweet(raw, "glove-v1") == expected


@pytest.mark.parametrize(
    "raw, expected",
    [
        # HTML entities, which the dataset carries verbatim.
        ("I can't stand this &amp; I won't lie", "i can not stand this & i will not lie"),
        ("&lt;3 you", "<3 you"),
        ("a &gt; b", "a > b"),
        ("&quot;quoted&quot;", '"quoted"'),
        # camelCase hashtags split on their case boundaries.
        ("#NotHappy today", "not happy today"),
        ("#iPhoneX", "i phone x"),
        ("#USAToday", "usa today"),
        # All-lowercase and all-caps hashtags are left as one word.
        ("#python is great", "python is great"),
        ("#BREAKING news", "breaking news"),
        ("#covid19", "covid19"),
        # Negation survives as the word "not".
        ("can't", "can not"),
        ("won't", "will not"),
        ("don't do it", "do not do it"),
        ("isn't it", "is not it"),
        ("wouldn't", "would not"),
        ("don\u2019t", "do not"),
        ("CAN'T", "can not"),
        # Everything glove-v1 did still happens.
        (
            "@user LOVED #ThisThing http://a.b \U0001f60d",
            "<user> loved this thing <url> smiling face with heart-eyes",
        ),
        ("mail jane@gmail.com please", "mail <email> please"),
        # A URL fragment is part of the URL, not a hashtag to split.
        ("see http://x.co/#SomeThing now", "see <url> now"),
    ],
)
def test_glove_v2(raw, expected):
    assert preprocess_tweet(raw, "glove-v2") == expected


@pytest.mark.parametrize(
    "raw",
    [
        "the meeting is at noon",
        "@bob see http://x.co",
        "love it \U0001f60d",
        "mail jane@gmail.com please",
        "#python is great",
        "  Multiple   spaces\nand CAPS  ",
    ],
)
def test_glove_v2_agrees_with_v1_where_nothing_new_applies(raw):
    assert preprocess_tweet(raw, "glove-v2") == preprocess_tweet(raw, "glove-v1")


def test_glove_v2_is_idempotent():
    once = preprocess_tweet(
        "@bob can't see http://x.co #NotHappy &amp; \U0001f525", "glove-v2"
    )
    assert preprocess_tweet(once, "glove-v2") == once


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("@someone I LOVE this http://t.co/x", "@user I LOVE this http"),
        ("see www.example.com", "see http"),
        ("&amp; &lt;3 you", "& <3 you"),
        # Cased, and emoji left as emoji: the byte-level BPE knows them.
        ("Love it \U0001f60d", "Love it \U0001f60d"),
        ("#NotHappy can't", "#NotHappy can't"),
        ("mail jane@gmail.com", "mail jane@gmail.com"),
        ("  spaced   out\n", "spaced out"),
    ],
)
def test_cardiff_v1(raw, expected):
    assert preprocess_tweet(raw, "cardiff-v1") == expected


# --- the spec an artifact records ------------------------------------------


@pytest.fixture
def model_dir(monkeypatch, tmp_path):
    monkeypatch.setitem(preprocessing.MODEL_DIRS, "lr", tmp_path)
    return tmp_path


def test_an_artifact_without_a_spec_file_uses_glove_v1(model_dir):
    # Every artifact trained before the file existed was trained with it.
    assert preprocessing.artifact_spec("lr") == "glove-v1"


def test_the_spec_round_trips_through_the_file(model_dir):
    path = preprocessing.write_artifact_spec("lr", "glove-v2")
    assert path == model_dir / "preprocessing.json"
    assert json.loads(path.read_text(encoding="utf-8")) == {"spec": "glove-v2"}
    assert preprocessing.artifact_spec("lr") == "glove-v2"


def test_writing_an_unknown_spec_is_refused(model_dir):
    with pytest.raises(ValueError):
        preprocessing.write_artifact_spec("lr", "glove-v9")
    assert not (model_dir / "preprocessing.json").exists()


def test_a_recorded_spec_this_code_does_not_know_is_rejected_at_load(model_dir):
    (model_dir / "preprocessing.json").write_text(json.dumps({"spec": "glove-v9"}))
    with pytest.raises(ValueError, match="glove-v9"):
        preprocessing.artifact_spec("lr")


def test_the_alias_resolves_to_the_same_spec_file():
    assert preprocessing.spec_path("bert") == preprocessing.spec_path("roberta")
