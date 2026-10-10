"""A sample of the served model's test errors, for tagging by hand.

Run it as a module, after ``python -m src.evaluate`` has saved predictions::

    python -m src.errors export      # writes reports/errors.csv (100 errors)
    python -m src.errors summarise   # counts the tags filled in by hand

Aggregate scores say how often the model is wrong, not why. ``export`` draws
a seeded, uniform sample of the test tweets the served transformer gets wrong,
so the sample's mix of mistakes matches the model's, and writes them with an
empty ``tag`` column. Tagging is done by a person reading each tweet; the
categories are in :data:`TAGS`. ``summarise`` then counts the tags per
confusion (gold -> predicted) for docs/error-analysis.md.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections import Counter

import numpy as np

from src import evaluate
from src.config import LABELS, REPORTS_DIR, SEED

LOGGER = logging.getLogger(__name__)

MODEL = "roberta"
SAMPLE_SIZE = 100
ERRORS_PATH = REPORTS_DIR / "errors.csv"
COLUMNS = ("index", "gold", "predicted", "confidence", "text", "tag", "notes")

# What a tagger may write in the ``tag`` column, one per row. Several of these
# mirror a behavioural check in src.behaviour, so a frequent tag here points
# at a check whose synthetic cases miss how the failure looks in real tweets.
TAGS = {
    "sarcasm": "says the opposite of what it means",
    "negation": "a negator the model ignored or misread (behaviour: negation)",
    "mixed": "positive and negative content in the same tweet",
    "neutral-boundary": "mild sentiment or news-like text; neutral vs not is arguable",
    "label-noise": "the gold label looks wrong to a careful reader",
    "emoji": "the sentiment is carried by emoji or emoticons (behaviour: emoji)",
    "context": "needs world knowledge or the thread the tweet replies to",
    "other": "none of the above; say what in notes",
}


def sample_errors(labels, predictions, size: int = SAMPLE_SIZE, seed: int = SEED):
    """Indices of up to ``size`` misclassified examples, sorted, seeded."""
    wrong = np.flatnonzero(np.asarray(labels) != np.asarray(predictions))
    rng = np.random.default_rng(seed)
    chosen = rng.choice(wrong, size=min(size, len(wrong)), replace=False)
    return np.sort(chosen)


def export(model: str = MODEL, size: int = SAMPLE_SIZE, dataset=None) -> int:
    from src.data import load_raw_dataset

    labels, probabilities = evaluate.load_predictions(model, "test")
    predictions = probabilities.argmax(axis=1)
    texts = (load_raw_dataset() if dataset is None else dataset)["test"]["text"]
    if len(texts) != len(labels):
        raise ValueError("saved predictions do not match the test split")

    chosen = sample_errors(labels, predictions, size)
    ERRORS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with ERRORS_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for index in chosen:
            writer.writerow(
                {
                    "index": int(index),
                    "gold": LABELS[labels[index]],
                    "predicted": LABELS[predictions[index]],
                    "confidence": f"{probabilities[index].max():.3f}",
                    "text": texts[index],
                    "tag": "",
                    "notes": "",
                }
            )
    LOGGER.info(
        "Wrote %d of %d errors to %s",
        len(chosen),
        (labels != predictions).sum(),
        ERRORS_PATH,
    )
    return len(chosen)


def summarise(path=None) -> dict:
    """Tag counts overall and per confusion; fails on a tag not in TAGS."""
    with (path or ERRORS_PATH).open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    untagged = [row["index"] for row in rows if not row["tag"].strip()]
    unknown = sorted({row["tag"] for row in rows if row["tag"].strip()} - set(TAGS))
    if unknown:
        raise ValueError(f"unknown tags {unknown}; use one of {sorted(TAGS)}")
    tagged = [row for row in rows if row["tag"].strip()]
    return {
        "rows": len(rows),
        "untagged": untagged,
        "tags": Counter(row["tag"] for row in tagged),
        "by_confusion": Counter(
            (f"{row['gold']} -> {row['predicted']}", row["tag"]) for row in tagged
        ),
    }


def markdown_summary(summary: dict) -> str:
    total = sum(summary["tags"].values()) or 1
    rows = ["| Tag | Count | Share |", "|---|---|---|"]
    for tag, count in summary["tags"].most_common():
        rows.append(f"| {tag} | {count} | {count / total:.0%} |")
    rows += ["", "| Confusion | Tag | Count |", "|---|---|---|"]
    for (confusion, tag), count in sorted(
        summary["by_confusion"].items(), key=lambda item: (-item[1], item[0])
    ):
        rows.append(f"| {confusion} | {tag} | {count} |")
    return "\n".join(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["export", "summarise"])
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.command == "export":
        return 0 if export() else 1
    summary = summarise()
    if summary["untagged"]:
        LOGGER.warning("%d rows still untagged", len(summary["untagged"]))
    print(markdown_summary(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
