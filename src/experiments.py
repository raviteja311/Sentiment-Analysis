"""Model-selection experiments, judged on the validation split only.

The class-weight, decision-bias and label-shift comparisons in MODEL_CARD.md
were once judged on the test split. A choice made by looking at test scores
is a choice fitted to the test set, and the test score reported afterwards is
no longer an estimate of anything. Every comparison here therefore reads
validation and nothing else; test is consulted once, for the final table.

Run it as a module::

    python -m src.experiments --models lstm gru
    python -m src.experiments --models lstm gru --experiments decision_bias label_shift

Each experiment writes ``reports/experiments/<experiment>_<model>.json`` and
the run prints one Markdown table for the model card.

* ``class_weight`` retrains the model with ``class_weight="balanced"`` into a
  scratch directory and compares its validation scores with the served
  artifact's. Recurrent models only; the linear baseline is already balanced.
* ``decision_bias`` fits an additive per-class offset on log-probabilities to
  maximise macro F1, two-fold within validation: fitted on one half and scored
  on the other, then swapped, so the offset is never judged on the examples
  it was fitted to.
* ``label_shift`` re-estimates the class prior of the validation split with
  the EM procedure of Saerens, Latinne and Decaestecker (2002), which uses no
  labels, and rescales the probabilities accordingly.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from datetime import UTC, datetime

import numpy as np
from sklearn.metrics import f1_score

from src.config import (
    LABELS,
    METRICS_DIR,
    MODEL_ALIASES,
    MODEL_KEYS,
    NUM_LABELS,
    REPORTS_DIR,
    SEED,
    SEQUENCE_CONFIG,
    SequenceConfig,
    resolve_model,
)
from src.utils.io import load_json, save_json
from src.utils.metrics import compute_metrics

LOGGER = logging.getLogger(__name__)

EXPERIMENTS: tuple[str, ...] = ("class_weight", "decision_bias", "label_shift")
EXPERIMENTS_DIR = REPORTS_DIR / "experiments"
# Scratch artifacts from variant retrains; gitignored, only the JSON is kept.
SCRATCH_DIR = EXPERIMENTS_DIR / "scratch"

EPSILON = 1e-12

# Only the models trained here with a class_weight setting can be re-weighted.
SEQUENCE_MODELS = ("lstm", "gru")


def scores(labels, predictions) -> dict:
    """The subset of compute_metrics a comparison is read from."""
    metrics = compute_metrics(labels, predictions)
    return {
        key: metrics[key]
        for key in ("accuracy", "f1_macro", "recall_macro", "f1_per_class")
    }


def delta(before: dict, after: dict) -> dict:
    return {
        "accuracy": after["accuracy"] - before["accuracy"],
        "f1_macro": after["f1_macro"] - before["f1_macro"],
        "recall_macro": after["recall_macro"] - before["recall_macro"],
        "f1_per_class": [
            a - b
            for a, b in zip(after["f1_per_class"], before["f1_per_class"], strict=True)
        ],
    }


def _mean_scores(parts: list[dict]) -> dict:
    return {
        "accuracy": float(np.mean([p["accuracy"] for p in parts])),
        "f1_macro": float(np.mean([p["f1_macro"] for p in parts])),
        "recall_macro": float(np.mean([p["recall_macro"] for p in parts])),
        "f1_per_class": np.mean([p["f1_per_class"] for p in parts], axis=0).tolist(),
    }


# ---------------------------------------------------------------------------
# Decision bias
# ---------------------------------------------------------------------------


def fit_decision_bias(
    probs: np.ndarray,
    labels: np.ndarray,
    grid: np.ndarray | None = None,
    sweeps: int = 2,
) -> np.ndarray:
    """Per-class offsets on log-probabilities that maximise macro F1.

    Coordinate ascent over a fixed grid, one class at a time: small enough to
    be exhaustive, deterministic, and impossible to overfit in interesting
    ways. The first offset is pinned by convention to keep the solution
    identifiable, since adding a constant to every class changes nothing.
    """
    grid = np.linspace(-1.5, 1.5, 31) if grid is None else grid
    log_probs = np.log(np.clip(probs, EPSILON, None))
    labels = np.asarray(labels)
    bias = np.zeros(NUM_LABELS)
    for _ in range(sweeps):
        for cls in range(1, NUM_LABELS):
            best_value, best_score = bias[cls], -1.0
            for value in grid:
                trial = bias.copy()
                trial[cls] = value
                score = f1_score(
                    labels, (log_probs + trial).argmax(axis=1), average="macro"
                )
                if score > best_score + 1e-12:
                    best_value, best_score = float(value), score
            bias[cls] = best_value
    return bias


def apply_decision_bias(probs: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return (np.log(np.clip(probs, EPSILON, None)) + bias).argmax(axis=1)


def decision_bias_experiment(
    probs: np.ndarray, labels: np.ndarray, seed: int = SEED
) -> dict:
    """Two-fold within validation: fit on one half, score on the other, swap."""
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(labels))
    half = len(order) // 2

    folds = []
    for fit_idx, eval_idx in ((order[:half], order[half:]), (order[half:], order[:half])):
        bias = fit_decision_bias(probs[fit_idx], labels[fit_idx])
        before = scores(labels[eval_idx], probs[eval_idx].argmax(axis=1))
        after = scores(labels[eval_idx], apply_decision_bias(probs[eval_idx], bias))
        folds.append({"bias": bias.tolist(), "before": before, "after": after})

    before = _mean_scores([fold["before"] for fold in folds])
    after = _mean_scores([fold["after"] for fold in folds])
    return {
        "experiment": "decision_bias",
        "method": "per-class log-probability offsets, coordinate ascent on macro F1",
        "judged_on": "validation, two-fold held out",
        "folds": folds,
        "before": before,
        "after": after,
        "delta": delta(before, after),
    }


# ---------------------------------------------------------------------------
# Label shift
# ---------------------------------------------------------------------------


def estimate_priors_em(
    probs: np.ndarray,
    train_priors: np.ndarray,
    iterations: int = 200,
    tolerance: float = 1e-8,
) -> np.ndarray:
    """Class prior of the data ``probs`` were computed on, by EM.

    Saerens et al. (2002): reweight each posterior by the ratio of the
    current prior estimate to the training prior, renormalise, and take the
    mean as the next estimate. Uses no labels.
    """
    train_priors = np.asarray(train_priors, dtype=float)
    priors = train_priors.copy()
    for _ in range(iterations):
        adjusted = probs * (priors / train_priors)
        adjusted /= adjusted.sum(axis=1, keepdims=True)
        updated = adjusted.mean(axis=0)
        if np.abs(updated - priors).max() < tolerance:
            priors = updated
            break
        priors = updated
    return priors


def apply_priors(probs: np.ndarray, train_priors, target_priors) -> np.ndarray:
    adjusted = probs * (np.asarray(target_priors) / np.asarray(train_priors))
    return adjusted / adjusted.sum(axis=1, keepdims=True)


def label_shift_experiment(probs: np.ndarray, labels: np.ndarray, train_priors) -> dict:
    labels = np.asarray(labels)
    train_priors = np.asarray(train_priors, dtype=float)
    estimated = estimate_priors_em(probs, train_priors)
    actual = np.bincount(labels, minlength=NUM_LABELS) / len(labels)

    before = scores(labels, probs.argmax(axis=1))
    after = scores(labels, apply_priors(probs, train_priors, estimated).argmax(axis=1))
    return {
        "experiment": "label_shift",
        "method": "EM prior re-estimation (Saerens et al., 2002), unsupervised",
        "judged_on": "validation",
        "train_priors": dict(zip(LABELS, train_priors.tolist(), strict=True)),
        "estimated_priors": dict(zip(LABELS, estimated.tolist(), strict=True)),
        # Recorded for the reader; never used by the estimate.
        "actual_priors": dict(zip(LABELS, actual.tolist(), strict=True)),
        "before": before,
        "after": after,
        "delta": delta(before, after),
    }


# ---------------------------------------------------------------------------
# Class weight
# ---------------------------------------------------------------------------


def _builder(model: str):
    if model == "lstm":
        from src.models.lstm_model import build_lstm

        return build_lstm
    from src.models.gru_model import build_gru

    return build_gru


def class_weight_experiment(
    model: str, config: SequenceConfig = SEQUENCE_CONFIG, strategy: str = "balanced"
) -> dict:
    """Retrain with class weights into scratch and compare on validation.

    The unweighted side is the served artifact's own training record, which
    the same code and seed produced; the weighted side is trained here.
    """
    from src.training.train_sequence import train_sequence_model

    served = load_json(METRICS_DIR / f"{model}.json")
    weighted = train_sequence_model(
        model,
        _builder(model),
        replace(config, class_weight=strategy),
        out_dir=SCRATCH_DIR / f"class_weight_{model}",
    )
    before = {
        key: served["metrics"]["validation"][key]
        for key in ("accuracy", "f1_macro", "recall_macro", "f1_per_class")
    }
    after = {
        key: weighted["metrics"]["validation"][key]
        for key in ("accuracy", "f1_macro", "recall_macro", "f1_per_class")
    }
    return {
        "experiment": "class_weight",
        "method": f'class_weight="{strategy}" against the served unweighted run',
        "judged_on": "validation",
        "served_class_weight": served["hyperparameters"].get("class_weight"),
        "before": before,
        "after": after,
        "delta": delta(before, after),
    }


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def validation_probabilities(model: str, dataset, batch_size: int = 64):
    """Served probabilities on validation, and the labels."""
    from src.inference.predictor import load_predictor

    predictor = load_predictor(model)
    texts = dataset["validation"]["text"]
    labels = np.asarray(dataset["validation"]["label"])
    chunks = [
        predictor.predict_proba(list(texts[start : start + batch_size]))
        for start in range(0, len(texts), batch_size)
    ]
    return np.vstack(chunks), labels


def train_priors(model: str) -> np.ndarray:
    distribution = load_json(METRICS_DIR / f"{model}.json")["dataset"][
        "class_distribution"
    ]
    counts = np.array([distribution["train"][label] for label in LABELS], dtype=float)
    return counts / counts.sum()


def run(
    models: list[str],
    experiments: tuple[str, ...] = EXPERIMENTS,
    batch_size: int = 64,
) -> list[dict]:
    from src.data import load_raw_dataset

    dataset = load_raw_dataset()
    results = []
    for model in models:
        model = resolve_model(model)
        probs, labels = validation_probabilities(model, dataset, batch_size)
        for experiment in experiments:
            if experiment == "class_weight":
                if model not in SEQUENCE_MODELS:
                    LOGGER.info(
                        "Skipping class_weight for %s: not a sequence model.", model
                    )
                    continue
                result = class_weight_experiment(model)
            elif experiment == "decision_bias":
                result = decision_bias_experiment(probs, labels)
            elif experiment == "label_shift":
                result = label_shift_experiment(probs, labels, train_priors(model))
            else:
                raise ValueError(
                    f"Unknown experiment {experiment!r}. Known: {EXPERIMENTS}."
                )

            result.update(
                model=model,
                run_at=datetime.now(UTC).isoformat(timespec="seconds"),
            )
            path = EXPERIMENTS_DIR / f"{experiment}_{model}.json"
            save_json(result, path)
            LOGGER.info(
                "%s / %s: macro F1 %.4f -> %.4f (%+.4f) -> %s",
                model,
                experiment,
                result["before"]["f1_macro"],
                result["after"]["f1_macro"],
                result["delta"]["f1_macro"],
                path,
            )
            results.append(result)
    return results


def load_results(models: list[str] | None = None) -> list[dict]:
    results = []
    for model in models or MODEL_KEYS:
        for experiment in EXPERIMENTS:
            path = EXPERIMENTS_DIR / f"{experiment}_{model}.json"
            if path.exists():
                results.append(load_json(path))
    return results


def _cell(before: float, after: float) -> str:
    return f"{before:.4f} to {after:.4f} ({after - before:+.4f})"


def markdown_table(results: list[dict]) -> str:
    """Before-to-after per metric, judged on validation, for the model card."""
    if not results:
        return "_No experiments recorded. Run `python -m src.experiments`._"
    rows = [
        "| Model | Experiment | Accuracy | Macro F1 | "
        + " | ".join(f"F1 {label}" for label in LABELS)
        + " |",
        "|---|---|---|---|" + "---|" * len(LABELS),
    ]
    for result in results:
        before, after = result["before"], result["after"]
        cells = [
            _cell(before["accuracy"], after["accuracy"]),
            _cell(before["f1_macro"], after["f1_macro"]),
            *[
                _cell(b, a)
                for b, a in zip(
                    before["f1_per_class"], after["f1_per_class"], strict=True
                )
            ],
        ]
        rows.append(
            f"| {result['model']} | {result['experiment']} | " + " | ".join(cells) + " |"
        )
    return "\n".join(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--models",
        nargs="+",
        choices=[*MODEL_KEYS, *MODEL_ALIASES],
        metavar="MODEL",
        default=list(SEQUENCE_MODELS),
    )
    parser.add_argument(
        "--experiments", nargs="+", choices=EXPERIMENTS, default=list(EXPERIMENTS)
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--table-only", action="store_true", help="print existing results only"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    results = (
        load_results(args.models)
        if args.table_only
        else run(args.models, tuple(args.experiments), args.batch_size)
    )
    print()
    print(markdown_table(results))
    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())
