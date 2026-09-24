"""Text normalisation shared by every model.

The same function runs at training time and at inference, so anything changed
here invalidates the trained artifacts and needs a retrain.

The placeholder tokens are deliberate: ``<url>`` and ``<user>`` are the
conventions GloVe Twitter was trained with, so they carry meaningful vectors
rather than being noise. The Keras tokenizer is configured to keep the angle
brackets (see ``src.training.train_sequence``); without that it would strip them
and a mention would look exactly like the ordinary word "user".
"""

import re

import emoji

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


def preprocess_tweet(text: str | None) -> str:
    """Normalise a tweet for every model in this project.

    Lowercases, replaces URLs, email addresses and mentions with placeholders,
    keeps the word of a hashtag, turns emoji into words and collapses
    whitespace.
    """
    if not isinstance(text, str):
        return ""

    text = text.strip().lower()
    text = URL_RE.sub(" <url> ", text)
    text = EMAIL_RE.sub(" <email> ", text)
    text = MENTION_RE.sub(" <user> ", text)
    text = HASHTAG_RE.sub(r"\1", text)

    try:
        # Words, not a single glued token: demojize gives ":smiling_face:", and
        # the underscores would survive tokenization as one long out-of-
        # vocabulary term. Split apart, each word has a pretrained vector.
        text = emoji.demojize(text, delimiters=(" ", " "))
        text = text.replace("_", " ")
    except Exception:
        pass

    return WHITESPACE_RE.sub(" ", text).strip()
