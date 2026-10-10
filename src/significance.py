"""How much the test scores would move, and which gaps are more than noise.

Run it as a module, after ``python -m src.evaluate`` has saved predictions::

    python -m src.significance              # every model with saved predictions
    python -m src.significance --table-only # re-print the tables, compute nothing

A single score on 12,284 tweets says nothing about its own precision, and a
1.2-point gap between two models can be real or a draw of the sample. This
module answers both from the saved per-example predictions:

* a 95% bootstrap confidence interval for macro F1 and macro recall, written
  into each model's metrics record;
* for pairs of models scored on the same tweets, McNemar's test on the
  examples where exactly one of the two is right, and a paired bootstrap
  interval for the difference in each metric, written to
  reports/experiments/significance.json.
"""

from __future__ import annotations

import argparse
import logging
import sys

import numpy as np
from scipy import stats

from src import evaluate
from src.config import LABELS, METRICS_DIR, REPORTS_DIR, SEED
from src.utils.io import load_json, save_json

LOGGER = logging.getLogger(__name__)

SPLIT = "test"
N_RESAMPLES = 1000
CONFIDENCE = 0.95
SIGNIFICANCE_PATH = REPORTS_DIR / "experiments" / "significance.json"

# The comparisons the model card makes in words. The served transformer
# against each base checkpoint it is measured against, and the two recurrent
# models it calls indistinguishable.
DEFAULT_PAIRS = (
    ("roberta", "roberta_base"),
    ("roberta", "roberta_base__twitter_roberta_base_sentiment_latest"),
    ("roberta_base", "roberta_base__twitter_roberta_base_sentiment_latest"),
    ("lstm", "gru"),
)

METRICS = ("f1_macro", "recall_macro")


def confusion_matrices(
    labels: np.ndarray, predictions: np.ndarray, samples
) -> np.ndarray:
    """One confusion matrix per row of ``samples`` (indices into the data).

    Shape ``(len(samples), k, k)``, true class by predicted class. Counting
    ``true * k + predicted`` with ``bincount`` builds each matrix in one pass,
    which keeps a thousand resamples of 12k tweets to well under a second.
    """
    k = len(LABELS)
    codes = labels * k + predictions
    return np.stack(
        [np.bincount(codes[rows], minlength=k * k).reshape(k, k) for rows in samples]
    )


def macro_scores(matrices: np.ndarray) -> dict[str, np.ndarray]:
    """Macro F1 and macro recall from a stack of confusion matrices.

    A class with no true or no predicted examples scores 0 for that term, as
    sklearn's ``zero_division=0`` does; on 12k tweets it never happens.
    """
    true_positive = np.diagonal(matrices, axis1=1, axis2=2).astype(float)
    actual = matrices.sum(axis=2)
    predicted = matrices.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.where(actual > 0, true_positive / actual, 0.0)
        precision = np.where(predicted > 0, true_positive / predicted, 0.0)
        f1 = np.where(
            precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0
        )
    return {"f1_macro": f1.mean(axis=1), "recall_macro": recall.mean(axis=1)}


def resample_indices(n: int, n_resamples: int = N_RESAMPLES, seed: int = SEED):
    """Bootstrap samples of ``n`` indices, drawn with replacement.

    Fixed by ``seed`` so the intervals in the README are reproducible, and
    shared between models in a pair so their difference is measured on the
    same resampled tweets.
    """
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(n_resamples, n))


def _interval(values: np.ndarray, confidence: float = CONFIDENCE) -> list[float]:
    tail = (1 - confidence) / 2 * 100
    low, high = np.percentile(values, [tail, 100 - tail])
    return [float(low), float(high)]


def bootstrap_intervals(labels, predictions, samples) -> dict:
    """Percentile confidence intervals for each metric in ``METRICS``."""
    scores = macro_scores(confusion_matrices(labels, predictions, samples))
    return {metric: _interval(scores[metric]) for metric in METRICS}


def mcnemar(labels, predictions_a, predictions_b) -> dict:
    """McNemar's exact test on the tweets exactly one of two models gets right.

    Tweets both models get right or both get wrong say nothing about which is
    better. Under the null hypothesis that the two are equally accurate, each
    of the ``n`` discordant tweets is equally likely to favour either model,
    so the count favouring A is Binomial(n, 0.5); the p-value is the two-sided
    tail of that distribution. The exact test is used rather than the
    chi-squared approximation because it needs no continuity correction and
    the two agree at these sample sizes anyway.
    """
    right_a = predictions_a == labels
    right_b = predictions_b == labels
    only_a = int(np.sum(right_a & ~right_b))
    only_b = int(np.sum(~right_a & right_b))
    discordant = only_a + only_b
    p_value = (
        float(stats.binomtest(only_a, discordant, 0.5).pvalue) if discordant else 1.0
    )
    return {"only_a_correct": only_a, "only_b_correct": only_b, "p_value": p_value}


def compare(model_a: str, model_b: str, split: str = SPLIT, samples=None) -> dict:
    """McNemar's test and paired bootstrap differences (A minus B)."""
    labels_a, probabilities_a = evaluate.load_predictions(model_a, split)
    labels_b, probabilities_b = evaluate.load_predictions(model_b, split)
    if not np.array_equal(labels_a, labels_b):
        raise ValueError(f"{model_a} and {model_b} were not scored on the same tweets")
    predictions_a = probabilities_a.argmax(axis=1)
    predictions_b = probabilities_b.argmax(axis=1)
    if samples is None:
        samples = resample_indices(len(labels_a))

    scores_a = macro_scores(confusion_matrices(labels_a, predictions_a, samples))
    scores_b = macro_scores(confusion_matrices(labels_a, predictions_b, samples))
    point_a = macro_scores(confusion_matrices(labels_a, predictions_a, [slice(None)]))
    point_b = macro_scores(confusion_matrices(labels_a, predictions_b, [slice(None)]))

    difference = {
        metric: {
            "estimate": float(point_a[metric][0] - point_b[metric][0]),
            "interval": _interval(scores_a[metric] - scores_b[metric]),
        }
        for metric in METRICS
    }
    return {
        "a": model_a,
        "b": model_b,
        "split": split,
        "mcnemar": mcnemar(labels_a, predictions_a, predictions_b),
        "difference": difference,
    }


def models_with_predictions(split: str = SPLIT) -> list[str]:
    return [
        report["model"]
        for report in evaluate.load_reports()
        if evaluate.predictions_path(report["model"], split).exists()
    ]


def record_intervals(model: str, split: str = SPLIT, samples=None) -> dict:
    """Add bootstrap intervals to a model's metrics record and save it."""
    labels, probabilities = evaluate.load_predictions(model, split)
    if samples is None:
        samples = resample_indices(len(labels))
    path = METRICS_DIR / f"{model}.json"
    record = load_json(path)
    record["metrics"][split]["bootstrap"] = {
        "n_resamples": len(samples),
        "confidence": CONFIDENCE,
        "seed": SEED,
        **bootstrap_intervals(labels, probabilities.argmax(axis=1), samples),
    }
    save_json(record, path)
    return record


def run(pairs=DEFAULT_PAIRS, split: str = SPLIT) -> dict:
    models = models_with_predictions(split)
    if not models:
        LOGGER.warning("No saved predictions. Run `python -m src.evaluate` first.")
        return {}
    for model in models:
        record_intervals(model, split)
        LOGGER.info("Bootstrap intervals recorded for %s", model)

    comparisons = []
    for model_a, model_b in pairs:
        if model_a in models and model_b in models:
            comparisons.append(compare(model_a, model_b, split))
        else:
            LOGGER.warning("Skipping %s vs %s: predictions missing", model_a, model_b)
    result = {
        "split": split,
        "n_resamples": N_RESAMPLES,
        "confidence": CONFIDENCE,
        "seed": SEED,
        "test": "McNemar exact (binomial on discordant pairs)",
        "comparisons": comparisons,
    }
    save_json(result, SIGNIFICANCE_PATH)
    return result


# --- tables -------------------------------------------------------------------


def _ci(interval) -> str:
    return f"{interval[0]:.4f}-{interval[1]:.4f}"


def interval_table(reports: list[dict], split: str = SPLIT) -> str:
    """Point scores with their bootstrap intervals, ranked as the main table."""
    rows = [
        "| Model | Macro F1 | 95% CI | Macro recall | 95% CI |",
        "|---|---|---|---|---|",
    ]
    for report in sorted(reports, key=evaluate._ranking_score):
        metrics = (report.get("metrics") or {}).get(split) or {}
        bootstrap = metrics.get("bootstrap")
        if bootstrap is None:
            continue
        rows.append(
            f"| {report.get('display_name', report['model'])} "
            f"| {metrics['f1_macro']:.4f} | {_ci(bootstrap['f1_macro'])} "
            f"| {evaluate.macro_recall(metrics):.4f} | {_ci(bootstrap['recall_macro'])} |"
        )
    return "\n".join(rows)


def comparison_table(result: dict) -> str:
    """Each pair's differences (A minus B) and McNemar p-value."""
    names = {
        r["model"]: r.get("display_name", r["model"]) for r in evaluate.load_reports()
    }
    rows = [
        "| A | B | Macro F1, A - B (95% CI) | Macro recall, A - B (95% CI) "
        "| Only A right | Only B right | McNemar p |",
        "|---|---|---|---|---|---|---|",
    ]
    for comparison in result.get("comparisons", []):
        difference = comparison["difference"]
        cells = [
            f"{difference[m]['estimate']:+.4f} ({difference[m]['interval'][0]:+.4f} "
            f"to {difference[m]['interval'][1]:+.4f})"
            for m in METRICS
        ]
        test = comparison["mcnemar"]
        rows.append(
            f"| {names.get(comparison['a'], comparison['a'])} "
            f"| {names.get(comparison['b'], comparison['b'])} "
            f"| {cells[0]} | {cells[1]} "
            f"| {test['only_a_correct']} | {test['only_b_correct']} "
            f"| {_p_value(test['p_value'])} |"
        )
    return "\n".join(rows)


def _p_value(p: float) -> str:
    return "<0.0001" if p < 1e-4 else f"{p:.4f}"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--table-only",
        action="store_true",
        help="print the tables from existing records without computing anything",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    result = load_json(SIGNIFICANCE_PATH) if args.table_only else run()
    if not result:
        return 1
    print()
    print(interval_table(evaluate.load_reports()))
    print()
    print(comparison_table(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
