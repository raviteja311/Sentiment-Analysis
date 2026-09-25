"""Behavioural checks, in the style of CheckList (Ribeiro et al., 2020).

Accuracy on a test split says how often a model is right; it says nothing
about *how* it is right. A model can score 0.71 while ignoring negation
entirely, because negated tweets are a minority of the split. The checks here
ask targeted questions instead: does "I don't like it" get a different label
from "I like it"; does swapping one username for another leave the label
alone; does a heart read as positive.

Run it as a module::

    python -m src.behaviour                  # every available model
    python -m src.behaviour --models lr gru

It prints a pass rate per check per model and writes
``reports/metrics/behaviour_<model>.json``. **It is a report, not a gate**:
model behaviour cannot be guaranteed by a test, only measured, so a failing
check never fails a build. The suite is small on purpose; it is there to be
read, and to catch a regression that the aggregate metrics would hide.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from src.config import LABELS, METRICS_DIR, MODEL_ALIASES, MODEL_KEYS, resolve_model
from src.utils.io import save_json

LOGGER = logging.getLogger(__name__)

# What a case expects of the labels. FLIP and SAME compare the two texts of a
# pair; anything else names the label the single text should get.
FLIP = "flip"
SAME = "same"


@dataclass(frozen=True)
class Case:
    text: str
    expect: str
    other: str | None = None


SUITE: dict[str, tuple[Case, ...]] = {
    # The label must change when the sentiment is negated.
    "negation flips the label": (
        Case("I like it", FLIP, "I don't like it"),
        Case("this is good", FLIP, "this is not good"),
        Case("I love this phone", FLIP, "I can't stand this phone"),
        Case("the service was great", FLIP, "the service wasn't great"),
        Case("I'm happy with the update", FLIP, "I'm not happy with the update"),
    ),
    # HTML entities the dataset carries verbatim.
    "entities read as what they encode": (
        Case("&lt;3 you", "positive"),
        Case("<3 this song", "positive"),
        Case("love you &amp; miss you", "positive"),
    ),
    "emoji carry sentiment": (
        Case("this is \U0001f60d", "positive"),
        Case("so happy \U0001f60a", "positive"),
        Case("ugh \U0001f62d", "negative"),
        Case("what a day \U0001f621", "negative"),
    ),
    # Who is mentioned must not change the label.
    "mention invariance": (
        Case("@alice thanks for the help", SAME, "@bob thanks for the help"),
        Case("@alice this is awful", SAME, "@zed this is awful"),
        Case("@support the app keeps crashing", SAME, "@helpdesk the app keeps crashing"),
    ),
    # Which link is shared must not change the label.
    "url invariance": (
        Case(
            "check this out http://a.co/x it's great",
            SAME,
            "check this out http://b.org/y it's great",
        ),
        Case(
            "worst update ever http://a.co",
            SAME,
            "worst update ever https://example.com/p",
        ),
        Case(
            "meeting notes are up http://x.io/1",
            SAME,
            "meeting notes are up http://y.io/2",
        ),
    ),
}

Labeller = Callable[[Sequence[str]], Sequence[str]]


def run_check(label: Labeller, cases: Sequence[Case]) -> dict:
    """Score one check; returns pass counts and every failing case."""
    texts = [case.text for case in cases]
    others = [case.other for case in cases if case.other is not None]
    labels = list(label(texts + others))
    got_text = labels[: len(texts)]
    got_other = iter(labels[len(texts) :])

    failures = []
    for case, got in zip(cases, got_text, strict=True):
        if case.expect in (FLIP, SAME):
            other = next(got_other)
            passed = (got != other) if case.expect == FLIP else (got == other)
            if not passed:
                failures.append(
                    {"text": case.text, "other": case.other, "labels": [got, other]}
                )
        else:
            if got != case.expect:
                failures.append(
                    {"text": case.text, "expected": case.expect, "label": got}
                )

    passed = len(cases) - len(failures)
    return {
        "passed": passed,
        "total": len(cases),
        "pass_rate": passed / len(cases) if cases else 1.0,
        "failures": failures,
    }


def run_suite(label: Labeller, suite: dict[str, Sequence[Case]] = SUITE) -> dict:
    """Every check against one labelling function."""
    checks = {name: run_check(label, cases) for name, cases in suite.items()}
    passed = sum(check["passed"] for check in checks.values())
    total = sum(check["total"] for check in checks.values())
    return {
        "checks": checks,
        "passed": passed,
        "total": total,
        "pass_rate": passed / total if total else 1.0,
    }


def labeller_for(model: str) -> Labeller:
    from src.inference.predictor import load_predictor

    predictor = load_predictor(model)
    return lambda texts: [p.label for p in predictor.predict_many(list(texts))]


def check_model(model: str) -> dict:
    """Run the suite against one served model and write its report."""
    from src.inference.predictor import load_predictor

    model = resolve_model(model)
    result = run_suite(labeller_for(model))
    result.update(
        model=model,
        version=load_predictor(model).version,
        checked_at=datetime.now(UTC).isoformat(timespec="seconds"),
        labels=list(LABELS),
    )
    save_json(result, report_path(model))
    return result


def report_path(model: str):
    return METRICS_DIR / f"behaviour_{resolve_model(model)}.json"


def markdown_table(results: Sequence[dict]) -> str:
    """Pass rate per check, one column per model."""
    if not results:
        return "_No models checked._"
    models = [result["model"] for result in results]
    rows = [
        "| Check | " + " | ".join(models) + " |",
        "|---|" + "---|" * len(models),
    ]
    for name in SUITE:
        cells = [
            f"{r['checks'][name]['passed']}/{r['checks'][name]['total']}" for r in results
        ]
        rows.append(f"| {name} | " + " | ".join(cells) + " |")
    rows.append(
        "| **all** | "
        + " | ".join(f"**{r['passed']}/{r['total']}**" for r in results)
        + " |"
    )
    return "\n".join(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--models",
        nargs="+",
        choices=[*MODEL_KEYS, *MODEL_ALIASES],
        metavar="MODEL",
        help="models to check (default: every available model)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from src.inference.predictor import ModelUnavailableError, available_models

    requested = (
        [resolve_model(m) for m in args.models] if args.models else available_models()
    )
    results = []
    for model in requested:
        try:
            result = check_model(model)
        except ModelUnavailableError as error:
            LOGGER.warning("Skipping %s: %s", model, error)
            continue
        LOGGER.info(
            "%s: %d/%d passed -> %s",
            model,
            result["passed"],
            result["total"],
            report_path(model),
        )
        for name, check in result["checks"].items():
            for failure in check["failures"]:
                LOGGER.info("  failed %s: %s", name, failure)
        results.append(result)

    print()
    print(markdown_table(results))
    # A report, not a gate: failing checks are findings, not build failures.
    return 0


if __name__ == "__main__":
    sys.exit(main())
