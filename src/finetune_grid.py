"""The Phase 2 fine-tune grid: train it, summarise it, apply the serving rule.

The runs, hyperparameters and serving rule are fixed in docs/serving-rule.md,
committed before any run was trained. This module only carries them out::

    python -m src.finetune_grid run                     # every run, every seed
    python -m src.finetune_grid run --runs B --seeds 13 # one cell
    python -m src.finetune_grid run --limit 64          # smoke test, not a result
    python -m src.finetune_grid summarise               # table + serving decision

``run`` is resumable: a cell whose record exists is skipped, so a GPU session
that times out loses only the cell in flight. Each cell writes its record,
training history and raw predictions (no weights) to
reports/experiments/finetune_grid/<run>_seed<seed>/.

``summarise`` needs the base checkpoint's test predictions, written by
``python -m src.evaluate --include-base``, and applies the rule only once every
candidate run has all of its seeds.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import asdict, dataclass, replace

import numpy as np

from src import evaluate
from src.calibration import apply_temperature, expected_calibration_error, fit_temperature
from src.config import BASELINE_KEY, REPORTS_DIR, ROBERTA_CONFIG, SEED
from src.significance import (
    CONFIDENCE,
    N_RESAMPLES,
    confusion_matrices,
    macro_scores,
    resample_indices,
)
from src.utils.io import load_json, save_json

LOGGER = logging.getLogger(__name__)

GRID_DIR = REPORTS_DIR / "experiments" / "finetune_grid"
# Smoke tests (--limit) land here, git-ignored and never summarised, so a
# truncated cell can never be mistaken for a finished one and skipped.
SMOKE_DIR = REPORTS_DIR / "experiments" / "scratch" / "finetune_grid_smoke"
SUMMARY_PATH = REPORTS_DIR / "experiments" / "finetune_grid.json"


@dataclass(frozen=True)
class GridRun:
    name: str
    base_model: str
    preprocessing: str
    question: str


GRID = {
    "A": GridRun(
        "A'",
        "cardiffnlp/twitter-roberta-base-sentiment",
        "glove-v1",
        "the served recipe under the grid's settings",
    ),
    "B": GridRun(
        "B",
        "cardiffnlp/twitter-roberta-base-sentiment",
        "cardiff-v1",
        "A' vs B: was the preprocessing mismatch the cause?",
    ),
    "C": GridRun(
        "C",
        "cardiffnlp/twitter-roberta-base",
        "cardiff-v1",
        "what fine-tuning adds from a model that never saw the labels",
    ),
    "D": GridRun(
        "D",
        "roberta-base",
        "cardiff-v1",
        "C vs D: what the Twitter pretraining is worth",
    ),
}
SEEDS = (13, 42, 1234)

# docs/serving-rule.md, "Fixed hyperparameters".
GRID_CONFIG = replace(
    ROBERTA_CONFIG,
    batch_size=32,
    epochs=3,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.06,
    max_len=128,
    early_stopping_patience=1,
)


def cell_dir(run: str, seed: int, root=None):
    return (root or GRID_DIR) / f"{run}_seed{seed}"


def is_done(run: str, seed: int, root=None) -> bool:
    return (cell_dir(run, seed, root) / "metrics.json").exists()


def train_cell(
    run: str, seed: int, limit: int | None = None, batch_size: int | None = None
):
    """Train one (run, seed) cell into its own directory."""
    from src.training.train_roberta import RunOptions, main

    spec = GRID[run]
    config = replace(GRID_CONFIG, base_model=spec.base_model)
    if batch_size is not None:
        config = replace(config, batch_size=batch_size)
    options = RunOptions(
        seed=seed,
        preprocessing=spec.preprocessing,
        select_on="dev",
        select_metric="recall_macro",
        save_weights=False,
        save_predictions=True,
        limit=limit,
    )
    LOGGER.info(
        "=== %s seed %d: %s, %s ===", spec.name, seed, spec.base_model, spec.preprocessing
    )
    root = SMOKE_DIR if limit is not None else GRID_DIR
    return main(config, out_dir=cell_dir(run, seed, root), options=options)


def run_grid(
    runs=None, seeds=SEEDS, limit=None, batch_size=None
) -> list[tuple[str, int]]:
    trained = []
    root = SMOKE_DIR if limit is not None else GRID_DIR
    for run in runs or GRID:
        for seed in seeds:
            if is_done(run, seed, root):
                LOGGER.info("Skipping %s seed %d: already trained", GRID[run].name, seed)
                continue
            train_cell(run, seed, limit=limit, batch_size=batch_size)
            trained.append((run, seed))
    return trained


# --- summarising ---------------------------------------------------------------


def _seed_scores(labels, predictions_by_seed, samples) -> dict[str, np.ndarray]:
    """Each metric per resample, averaged over seeds inside the resample."""
    per_seed = [
        macro_scores(confusion_matrices(labels, predictions, samples))
        for predictions in predictions_by_seed
    ]
    return {
        metric: np.mean([scores[metric] for scores in per_seed], axis=0)
        for metric in per_seed[0]
    }


def _interval(values) -> list[float]:
    tail = (1 - CONFIDENCE) / 2 * 100
    low, high = np.percentile(values, [tail, 100 - tail])
    return [float(low), float(high)]


def _mean_std(values) -> dict:
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
    }


def summarise_run(run: str, samples, test_labels) -> dict | None:
    seeds = [seed for seed in SEEDS if is_done(run, seed)]
    if not seeds:
        return None
    records, test_predictions, calibration = [], [], []
    for seed in seeds:
        directory = cell_dir(run, seed)
        records.append(load_json(directory / "metrics.json"))
        labels, test_raw, _ = evaluate.read_predictions(
            directory / "predictions" / "test.npz"
        )
        if not np.array_equal(labels, test_labels):
            raise ValueError(
                f"{run} seed {seed} was not scored on the base's test tweets"
            )
        test_predictions.append(test_raw.argmax(axis=1))
        dev_labels, dev_raw, _ = evaluate.read_predictions(
            directory / "predictions" / "dev.npz"
        )
        temperature = fit_temperature(dev_raw, dev_labels)
        calibration.append(
            {
                "temperature": temperature,
                "ece_before": expected_calibration_error(test_raw, labels),
                "ece_after": expected_calibration_error(
                    apply_temperature(test_raw, temperature), labels
                ),
            }
        )
    limits = {record["hyperparameters"].get("limit") for record in records}
    scores = _seed_scores(test_labels, test_predictions, samples)
    return {
        "run": run,
        "name": GRID[run].name,
        **{k: v for k, v in asdict(GRID[run]).items() if k != "name"},
        "seeds": seeds,
        "complete": len(seeds) == len(SEEDS) and limits == {None},
        "best_dev_seed": max(
            seeds, key=lambda s: records[seeds.index(s)]["metrics"]["dev"]["recall_macro"]
        ),
        "test": {
            metric: _mean_std([r["metrics"]["test"][metric] for r in records])
            for metric in ("recall_macro", "f1_macro", "accuracy")
        },
        "dev_recall_macro": _mean_std(
            [r["metrics"]["dev"]["recall_macro"] for r in records]
        ),
        "test_interval": {metric: _interval(scores[metric]) for metric in scores},
        "calibration": {
            key: _mean_std([c[key] for c in calibration])
            for key in ("temperature", "ece_before", "ece_after")
        },
        "_scores": scores,
    }


def apply_rule(base: dict, runs: list[dict]) -> dict:
    """docs/serving-rule.md, steps 1-4."""
    if not runs or not all(run["complete"] for run in runs) or len(runs) < len(GRID):
        return {"decision": "incomplete", "reason": "not every run has all its seeds"}
    winner = max(runs, key=lambda run: run["test"]["recall_macro"]["mean"])
    if winner["test"]["recall_macro"]["mean"] <= base["recall_macro"]:
        return {
            "decision": "serve base checkpoint",
            "winner": "base",
            "reason": "no run beat it",
        }
    gain = winner["_scores"]["recall_macro"] - base["_scores"]["recall_macro"]
    interval = _interval(gain)
    serve_winner = interval[0] > 0
    return {
        "decision": (
            f"serve {winner['name']} seed {winner['best_dev_seed']}"
            if serve_winner
            else "serve base checkpoint"
        ),
        "winner": winner["name"],
        "gain_over_base": {
            "estimate": winner["test"]["recall_macro"]["mean"] - base["recall_macro"],
            "interval": interval,
        },
        "reason": (
            "the gain's interval excludes zero"
            if serve_winner
            else "the gain's interval includes zero, so the simpler base is served"
        ),
    }


def summarise() -> dict:
    labels, base_raw, _ = evaluate.load_raw_predictions(BASELINE_KEY, "test")
    samples = resample_indices(len(labels), N_RESAMPLES, SEED)
    base_predictions = base_raw.argmax(axis=1)
    base_scores = macro_scores(confusion_matrices(labels, base_predictions, samples))
    point = macro_scores(confusion_matrices(labels, base_predictions, [slice(None)]))
    base = {
        "recall_macro": float(point["recall_macro"][0]),
        "f1_macro": float(point["f1_macro"][0]),
        "accuracy": float((base_predictions == labels).mean()),
        "_scores": base_scores,
    }
    runs = [r for r in (summarise_run(run, samples, labels) for run in GRID) if r]
    decision = apply_rule(base, runs)
    strip = lambda d: {k: v for k, v in d.items() if not k.startswith("_")}  # noqa: E731
    result = {
        "rule": "docs/serving-rule.md",
        "seeds": list(SEEDS),
        "hyperparameters": asdict(GRID_CONFIG) | {"select_on": "dev"},
        "base_checkpoint": strip(base),
        "runs": [strip(run) for run in runs],
        "decision": decision,
    }
    save_json(result, SUMMARY_PATH)
    return result


def _ms(stat: dict) -> str:
    return f"{stat['mean']:.4f} ± {stat['std']:.4f}"


def markdown_table(result: dict) -> str:
    base = result["base_checkpoint"]
    rows = [
        "| Run | Start | Preprocessing | Seeds | Test macro recall | 95% CI "
        "| Test macro F1 | Fitted T | Raw test ECE |",
        "|---|---|---|---|---|---|---|---|---|",
        f"| base | {GRID['B'].base_model} (as published) | cardiff-v1 | - "
        f"| {base['recall_macro']:.4f} | - | {base['f1_macro']:.4f} | - | - |",
    ]
    for run in result["runs"]:
        low, high = run["test_interval"]["recall_macro"]
        rows.append(
            f"| {run['name']} | {run['base_model']} | {run['preprocessing']} "
            f"| {len(run['seeds'])} | {_ms(run['test']['recall_macro'])} "
            f"| {low:.4f}-{high:.4f} | {_ms(run['test']['f1_macro'])} "
            f"| {run['calibration']['temperature']['mean']:.2f} "
            f"| {run['calibration']['ece_before']['mean']:.4f} |"
        )
    return "\n".join(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("run", help="train grid cells, skipping finished ones")
    train.add_argument("--runs", nargs="+", choices=list(GRID))
    train.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    train.add_argument("--limit", type=int, default=None, help="smoke test on N examples")
    train.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="override the fixed 32 only to fit a small GPU; recorded in each record",
    )
    sub.add_parser("summarise", help="aggregate finished cells and apply the rule")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.command == "run":
        run_grid(args.runs, args.seeds, args.limit, args.batch_size)
        return 0
    result = summarise()
    print(markdown_table(result))
    print()
    print(f"Decision: {result['decision']['decision']} ({result['decision']['reason']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
