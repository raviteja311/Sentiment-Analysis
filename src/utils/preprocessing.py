"""Text normalisation shared by every model, in versioned specs.

The same function runs at training time and at inference, so anything changed
here changes what a model sees. Preprocessing is therefore versioned: a spec is
frozen once an artifact has been trained with it, and every artifact records
the spec it was trained with in ``preprocessing.json`` beside its weights (see
:func:`artifact_spec`). New behaviour goes into a new spec, and the shipped
artifacts keep receiving the text they were trained on until they are
retrained.

* ``glove-v1`` is the original behaviour, kept byte-for-byte. Every artifact
  trained before specs existed uses it, which is why it is the default.
* ``glove-v2`` adds HTML unescaping, camelCase hashtag splitting and negation
  expansion. Tweets in the dataset carry entities, so ``&amp;`` reached the
  models as the word "amp" and ``&lt;3`` lost its heart; and both the TF-IDF
  and the Keras tokenizer split on the apostrophe, so ``can't`` arrived as
  "can" with the negation gone.
* ``cardiff-v1`` is for the transformer and matches what its base checkpoint,
  ``cardiffnlp/twitter-roberta-base-sentiment``, was trained on: mentions
  become ``@user``, URLs become ``http``, nothing else. No lowercasing and no
  emoji-to-words, because the byte-level BPE vocabulary is cased and already
  knows emoji.

The placeholder tokens in the GloVe specs are deliberate: ``<url>`` and
``<user>`` are the conventions GloVe Twitter was trained with, so they carry
meaningful vectors rather than being noise. The Keras tokenizer is configured
to keep the angle brackets (see ``src.training.train_sequence``); without that
it would strip them and a mention would look exactly like the ordinary word
"user".
"""

from __future__ import annotations

import html
import json
import re
from collections.abc import Callable
from pathlib import Path

import emoji

from src.config import MODEL_DIRS, resolve_model

# Ordered: URLs first, because a URL can contain an @ that would otherwise be
# mistaken for a mention.
URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)

# An address is not a mention. `@\w+` alone turned "jane@gmail.com" into
# "jane <user> .com", inventing a mention and mangling the rest.
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")

# A mention's @ must not follow a word character, or the local part of anything
# address-shaped gets swallowed.
MENTION_RE = re.compile(r"(?<![\w.])@\w+")

HASHTAG_RE = re.compile(r"#(\w+)")

WHITESPACE_RE = re.compile(r"\s+")

# Boundaries inside a camelCase word: "NotHappy" -> "Not Happy", "iPhoneX" ->
# "i Phone X", "USAToday" -> "USA Today".
CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")

# Typographic apostrophes, which phones insert by default, folded to the plain
# one so that one negation rule covers both.
CURLY_APOSTROPHES = str.maketrans({"\u2019": "'", "\u2018": "'", "\u02bc": "'"})
CANT_RE = re.compile(r"\bcan't\b")
WONT_RE = re.compile(r"\bwon't\b")
NT_RE = re.compile(r"n't\b")

DEFAULT_SPEC = "glove-v1"
PREPROCESSING_FILE = "preprocessing.json"


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


def _demojize(text: str) -> str:
    """Emoji as words, not as one glued token.

    demojize gives ":smiling_face:", and the underscores would survive
    tokenization as one long out-of-vocabulary term. Split apart, each word has
    a pretrained vector.
    """
    try:
        text = emoji.demojize(text, delimiters=(" ", " "))
        return text.replace("_", " ")
    except Exception:
        return text


def _split_hashtag(match: re.Match) -> str:
    word = match.group(1)
    # All-lowercase and all-caps hashtags have no case boundaries to split on
    # and are kept as one word rather than guessed at.
    if word.islower() or word.isupper():
        return word
    return CAMEL_BOUNDARY_RE.sub(" ", word)


def _expand_negations(text: str) -> str:
    """``can't`` to ``can not``, ``won't`` to ``will not``, then any ``n't``."""
    text = text.translate(CURLY_APOSTROPHES)
    text = CANT_RE.sub("can not", text)
    text = WONT_RE.sub("will not", text)
    return NT_RE.sub(" not", text)


# ---------------------------------------------------------------------------
# Specs
# ---------------------------------------------------------------------------


def _glove_v1(text: str) -> str:
    """The original normalisation. Frozen: the shipped artifacts depend on it."""
    text = text.strip().lower()
    text = URL_RE.sub(" <url> ", text)
    text = EMAIL_RE.sub(" <email> ", text)
    text = MENTION_RE.sub(" <user> ", text)
    text = HASHTAG_RE.sub(r"\1", text)
    text = _demojize(text)
    return WHITESPACE_RE.sub(" ", text).strip()


def _glove_v2(text: str) -> str:
    """glove-v1 plus entities, camelCase hashtags and negation."""
    # Entities first: "&lt;3" is only a heart once it is "<3", and "&amp;"
    # would otherwise tokenize as the word "amp".
    text = html.unescape(text).strip()
    # Placeholders before lowercasing, so the hashtag step still sees the case
    # boundaries it splits on. The patterns are case-insensitive, so this is
    # the same result glove-v1 gets from lowercasing first.
    text = URL_RE.sub(" <url> ", text)
    text = EMAIL_RE.sub(" <email> ", text)
    text = MENTION_RE.sub(" <user> ", text)
    text = HASHTAG_RE.sub(_split_hashtag, text)
    text = text.lower()
    text = _expand_negations(text)
    text = _demojize(text)
    return WHITESPACE_RE.sub(" ", text).strip()


def _cardiff_v1(text: str) -> str:
    """What the Twitter-RoBERTa base checkpoint was trained on."""
    text = html.unescape(text).strip()
    text = URL_RE.sub(" http ", text)
    text = MENTION_RE.sub(" @user ", text)
    return WHITESPACE_RE.sub(" ", text).strip()


SPECS: dict[str, Callable[[str], str]] = {
    "glove-v1": _glove_v1,
    "glove-v2": _glove_v2,
    "cardiff-v1": _cardiff_v1,
}


def resolve_spec(spec: str) -> Callable[[str], str]:
    """The normalisation function for a spec, or a ValueError naming the known ones."""
    try:
        return SPECS[spec]
    except KeyError:
        known = ", ".join(SPECS)
        raise ValueError(
            f"Unknown preprocessing spec {spec!r}. Known specs: {known}."
        ) from None


def preprocess_tweet(text: str | None, spec: str = DEFAULT_SPEC) -> str:
    """Normalise a tweet according to ``spec``.

    The default is ``glove-v1``, the spec every artifact trained before specs
    existed was trained with. Callers serving a model should pass the spec the
    artifact records (see :func:`artifact_spec`) rather than rely on it.
    """
    normalise = resolve_spec(spec)
    if not isinstance(text, str):
        return ""
    return normalise(text)


# ---------------------------------------------------------------------------
# The spec an artifact was trained with
# ---------------------------------------------------------------------------


def spec_path(model: str) -> Path:
    return MODEL_DIRS[resolve_model(model)] / PREPROCESSING_FILE


def artifact_spec(model: str) -> str:
    """The spec a model's artifact was trained with.

    Read from ``preprocessing.json`` beside the weights. An absent file means
    ``glove-v1``: every artifact trained before the file existed used it.
    """
    path = spec_path(model)
    if not path.exists():
        return DEFAULT_SPEC
    with open(path, encoding="utf-8") as handle:
        spec = json.load(handle)["spec"]
    # Fail when the model is loaded, not on its first prediction.
    resolve_spec(spec)
    return spec


def write_artifact_spec(model: str, spec: str, directory: Path | None = None) -> Path:
    """Record the spec beside the weights. Training calls this.

    ``directory`` overrides the model's directory, for training runs that
    must not touch the served artifacts (see src/experiments.py).
    """
    resolve_spec(spec)
    path = spec_path(model) if directory is None else Path(directory) / PREPROCESSING_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump({"spec": spec}, handle, indent=2)
        handle.write("\n")
    return path
