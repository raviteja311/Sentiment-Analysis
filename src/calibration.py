"""Temperature scaling, so that a confidence score means something.

Every model here reports a raw softmax (or ``predict_proba``) output as its
confidence. Those numbers are systematically overconfident - the transformer
routinely returns >0.99 on short inputs while being right about 71% of the time -
because nothing in the training objective rewards a model for knowing when it is
unsure. Presenting that through an API as "confidence" invites a reader to
threshold on it, and the threshold will not mean what they think.

Temperature scaling (Guo et al., 2017) fixes the scale with a single parameter
per model, fitted by minimising negative log-likelihood on the validation split::

    calibrated = softmax(log(p) / T)

The fitted temperature is constrained to T >= 1, so calibration can only ever
soften confidence. See :func:`fit_temperature` for why.

Two properties make it the right tool here:

* **It cannot change a prediction.** Dividing logits by a positive scalar is
  monotonic, so the arg max - and therefore every accuracy and F1 in the
  results table - is untouched. Only the spread of the probabilities moves.
* **It needs one parameter**, fitted on 2,000 held-out examples, so it cannot
  meaningfully overfit the way per-class isotonic regression could.

``log(p)`` recovers the logits up to an additive constant, and softmax is
shift-invariant, so working from probabilities is exact rather than an
approximation - which is what lets the same code calibrate the scikit-learn
pipeline and the neural models alike.

Fit and inspect with::

    make calibrate
    python -m src.calibration --models roberta
"""

from __future__ import annotations

import argparse
import logging
import sys

import numpy as np

from src.config import (
    MODEL_ALIASES,
    MODEL_DIRS,
    MODEL_KEYS,
    NUM_LABELS,
    SEED,
    resolve_model,
)
from src.utils.io import load_json, save_json

LOGGER = logging.getLogger(__name__)

# Bins used to measure calibration error. 15 is the usual choice in the
# literature; the result is not sensitive to it within a few bins either way.
DEFAULT_BINS = 15

# Guards log(0) for a model that assigns a class exactly zero probability.
EPSILON = 1e-12


def calibration_path(model: str):
    """Where a model's fitted temperature is stored."""
    return MODEL_DIRS[resolve_model(model)] / "calibration.json"


def load_temperature(model: str) -> float | None:
    """Fitted temperature for a model, or None if it has not been calibrated."""
    path = calibration_path(model)
    if not path.exists():
        return None
    try:
        return float(load_json(path)["temperature"])
    except (KeyError, ValueError, TypeError, OSError):
        LOGGER.warning("Ignoring unreadable calibration file: %s", path)
        return None


def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    """Rescale probabilities by ``temperature``.

    T > 1 softens an overconfident model; T < 1 sharpens an underconfident one;
    T == 1 is a no-op.
    """
    if temperature == 1.0:
        return probs
    logits = np.log(np.clip(probs, EPSILON, None)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Measures
# ---------------------------------------------------------------------------


def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = DEFAULT_BINS
) -> float:
    """Average gap between confidence and accuracy, weighted by bin population.

    0 is perfect: in every confidence band, the model is right exactly as often
    as it claims.
    """
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == np.asarray(labels)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    error = 0.0
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        in_bin = (confidence > low) & (confidence <= high)
        if not in_bin.any():
            continue
        error += in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(error)


def negative_log_likelihood(probs: np.ndarray, labels: np.ndarray) -> float:
    """Mean NLL of the true class. This is what temperature is fitted on."""
    rows = np.arange(len(labels))
    return float(-np.log(np.clip(probs[rows, np.asarray(labels)], EPSILON, None)).mean())


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Multi-class Brier score: mean squared error against the one-hot truth."""
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(labels)), np.asarray(labels)] = 1.0
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def mean_confidence(probs: np.ndarray) -> float:
    return float(probs.max(axis=1).mean())


def accuracy(probs: np.ndarray, labels: np.ndarray) -> float:
    return float((probs.argmax(axis=1) == np.asarray(labels)).mean())


def measure(probs: np.ndarray, labels: np.ndarray) -> dict:
    """Every calibration statistic for one set of predictions."""
    return {
        "ece": expected_calibration_error(probs, labels),
        "nll": negative_log_likelihood(probs, labels),
        "brier": brier_score(probs, labels),
        "mean_confidence": mean_confidence(probs),
        "accuracy": accuracy(probs, labels),
    }


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def fit_temperature(
    probs: np.ndarray, labels: np.ndarray, bounds: tuple[float, float] = (1.0, 10.0)
) -> float:
    """Temperature minimising NLL on the given predictions.

    A bounded scalar minimisation: NLL as a function of T is convex for
    temperature scaling, so there is one minimum and no starting point to choose.

    The lower bound is 1.0 on purpose - this softens, it never sharpens.
    Sharpening is only correct for a model that is *under*-confident, and the
    validation split here scores 3-9 points above test, so a model can look
    underconfident on validation purely because that split is easier. Fitting an
    unconstrained temperature on it does exactly that: the linear baseline fits
    T = 0.94 on validation, which then makes its calibration worse on test.
    Softening a model that turns out to be well calibrated costs little;
    sharpening one that is not is how you end up overstating confidence.
    """
    from scipy.optimize import minimize_scalar

    def objective(temperature: float) -> float:
        return negative_log_likelihood(apply_temperature(probs, temperature), labels)

    result = minimize_scalar(objective, bounds=bounds, method="bounded")
    return float(result.x)


def held_out_improvement(
    probs: np.ndarray, labels: np.ndarray, seed: int = SEED
) -> float:
    """Mean ECE gain from temperature scaling, measured within validation.

    Two-fold: fit on one half, measure on the other, then swap. This answers
    "does calibration help on data it did not see?" using only the validation
    split, so a model can be excluded from calibration without ever consulting
    the test set. Deciding that from the test numbers would be selecting on the
    held-out data, which is the failure this project exists to correct.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(labels))
    half = len(order) // 2

    gains = []
    folds = ((order[:half], order[half:]), (order[half:], order[:half]))
    for fit_idx, eval_idx in folds:
        temperature = fit_temperature(probs[fit_idx], labels[fit_idx])
        before = expected_calibration_error(probs[eval_idx], labels[eval_idx])
        after = expected_calibration_error(
            apply_temperature(probs[eval_idx], temperature), labels[eval_idx]
        )
        gains.append(before - after)
    return float(np.mean(gains))


def calibrate_model(model: str, dataset=None, batch_size: int = 64) -> dict:
    """Fit a temperature on validation and report its effect on test.

    Fitting and measurement use different splits on purpose: a temperature that
    only improves the data it was fitted on has told us nothing.
    """
    from src.data import load_raw_dataset
    from src.inference.predictor import load_predictor

    dataset = load_raw_dataset() if dataset is None else dataset
    predictor = load_predictor(model)

    def probabilities(split: str) -> tuple[np.ndarray, np.ndarray]:
        texts = dataset[split]["text"]
        labels = np.asarray(dataset[split]["label"])
        chunks = [
            predictor.predict_proba(list(texts[start : start + batch_size]))
            for start in range(0, len(texts), batch_size)
        ]
        raw = np.vstack(chunks) if chunks else np.zeros((0, NUM_LABELS))
        return raw, labels

    LOGGER.info("Calibrating %s...", model)

    # Uncalibrated probabilities: temporarily ignore any temperature already in
    # effect, so refitting does not compound on a previous fit.
    previous, predictor.temperature = predictor.temperature, 1.0
    try:
        val_probs, val_labels = probabilities("validation")
        test_probs, test_labels = probabilities("test")
    finally:
        predictor.temperature = previous

    candidate = fit_temperature(val_probs, val_labels)

    # Adopt the temperature only if it helps on validation data it was not
    # fitted to. A model that is already calibrated - the linear baseline is
    # close - gains nothing, and a temperature fitted on a 2,000-example split
    # that scores several points above test can make it worse.
    gain = held_out_improvement(val_probs, val_labels)
    adopted = gain > 0 and abs(candidate - 1.0) > 1e-3
    temperature = candidate if adopted else 1.0

    before = measure(test_probs, test_labels)
    after = measure(apply_temperature(test_probs, temperature), test_labels)

    if adopted:
        LOGGER.info(
            "  T=%.3f  ECE %.4f -> %.4f  NLL %.4f -> %.4f  mean confidence %.3f -> %.3f",
            temperature,
            before["ece"],
            after["ece"],
            before["nll"],
            after["nll"],
            before["mean_confidence"],
            after["mean_confidence"],
        )
    else:
        LOGGER.info(
            "  already calibrated (candidate T=%.3f gained %.4f ECE within "
            "validation); leaving it uncalibrated at T=1.0",
            candidate,
            gain,
        )

    record = {
        "model": model,
        "temperature": temperature,
        "candidate_temperature": candidate,
        "adopted": adopted,
        "held_out_ece_gain": gain,
        "method": "temperature scaling",
        "fitted_on": "validation",
        "measured_on": "test",
        "before": before,
        "after": after,
    }
    save_json(record, calibration_path(model))
    return record


def calibrate(models: list[str] | None = None, batch_size: int = 64) -> list[dict]:
    """Calibrate several models, skipping any that are unavailable."""
    from src.data import load_raw_dataset
    from src.inference.predictor import available_models, check_artifacts

    requested = [resolve_model(m) for m in models] if models else available_models()

    # Filter before touching the dataset. Skipping rather than crashing matches
    # src.evaluate - asking to calibrate a model whose weights are absent is a
    # normal thing to do on a fresh clone - and doing it first avoids
    # downloading tweet_eval only to discover there is nothing to calibrate.
    usable = []
    for model in requested:
        reason = check_artifacts(model)
        if reason is not None:
            LOGGER.warning("Skipping %s: %s", model, reason)
            continue
        usable.append(model)

    if not usable:
        LOGGER.warning("No usable models. Run `make fetch-weights` first.")
        return []

    dataset = load_raw_dataset()
    return [calibrate_model(model, dataset, batch_size) for model in usable]


def markdown_table(records: list[dict]) -> str:
    """Before/after table for the README and model card."""
    if not records:
        return "_No models calibrated yet. Run `make calibrate`._"

    rows = [
        "| Model | Temperature | ECE before | ECE after | Mean confidence before"
        " | after | Accuracy |",
        "|---|---|---|---|---|---|---|",
    ]
    for record in sorted(records, key=lambda item: -item["after"]["accuracy"]):
        temperature = (
            f"{record['temperature']:.2f}"
            if record.get("adopted", True)
            else "1.00 (not adopted)"
        )
        rows.append(
            "| {model} | {t} | {eb:.4f} | {ea:.4f} | {cb:.3f} | {ca:.3f} |"
            " {acc:.4f} |".format(
                model=record["model"],
                t=temperature,
                eb=record["before"]["ece"],
                ea=record["after"]["ece"],
                cb=record["before"]["mean_confidence"],
                ca=record["after"]["mean_confidence"],
                acc=record["after"]["accuracy"],
            )
        )
    return "\n".join(rows)


def load_records(models: list[str] | None = None) -> list[dict]:
    records = []
    for model in models or MODEL_KEYS:
        path = calibration_path(model)
        if path.exists():
            records.append(load_json(path))
    return records


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--models",
        nargs="+",
        # Aliases are documented, so the CLI has to take them too.
        choices=[*MODEL_KEYS, *MODEL_ALIASES],
        metavar="MODEL",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--table-only", action="store_true", help="print existing results only"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    records = (
        load_records(args.models)
        if args.table_only
        else calibrate(args.models, args.batch_size)
    )

    print()
    print(markdown_table(records))
    return 0 if records else 1


if __name__ == "__main__":
    sys.exit(main())
