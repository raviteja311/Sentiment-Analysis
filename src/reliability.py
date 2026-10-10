"""Reliability diagrams and a temperature for every model, base checkpoints too.

Run it as a module, after ``python -m src.evaluate`` has saved validation and
test predictions::

    python -m src.reliability

:mod:`src.calibration` fits the temperature each served model applies. This
module asks a different question of the same method: how overconfident is
each model, the untouched base checkpoints included, and does fine-tuning
change that? For every model with saved predictions it fits a temperature on
validation (as src.calibration does), measures expected calibration error on
test before and after, writes reports/experiments/calibration_comparison.json,
and draws reports/figures/reliability_<model>.svg.

Temperatures are fitted on validation, not on the dev split: every committed
model was trained on the dev tweets, so their confidence there is inflated.
Runs trained with the dev split held out should be fitted on it instead.
"""

from __future__ import annotations

import argparse
import logging
import sys
from html import escape

import numpy as np

from src import evaluate
from src.calibration import (
    DEFAULT_BINS,
    apply_temperature,
    expected_calibration_error,
    fit_temperature,
    held_out_improvement,
)
from src.config import REPORTS_DIR
from src.utils.io import load_json, save_json

LOGGER = logging.getLogger(__name__)

COMPARISON_PATH = REPORTS_DIR / "experiments" / "calibration_comparison.json"
FIGURES_DIR = REPORTS_DIR / "figures"


def reliability_bins(probs: np.ndarray, labels: np.ndarray, n_bins: int = DEFAULT_BINS):
    """Per confidence bin: mean confidence, accuracy and count; empty bins dropped.

    The same bins :func:`src.calibration.expected_calibration_error` sums
    over, so the diagram shows exactly the gaps that make up the ECE.
    """
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == labels
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        in_bin = (confidence > low) & (confidence <= high)
        if in_bin.any():
            bins.append(
                {
                    "confidence": float(confidence[in_bin].mean()),
                    "accuracy": float(correct[in_bin].mean()),
                    "count": int(in_bin.sum()),
                }
            )
    return bins


def models_with_both_splits() -> list[str]:
    return [
        report["model"]
        for report in evaluate.load_reports()
        if all(
            evaluate.predictions_path(report["model"], split).exists()
            for split in ("validation", "test")
        )
    ]


def compare_model(model: str) -> dict:
    """Fit on validation, measure on test, from the raw saved predictions."""
    val_labels, val_raw, served = evaluate.load_raw_predictions(model, "validation")
    test_labels, test_raw, _ = evaluate.load_raw_predictions(model, "test")

    temperature = fit_temperature(val_raw, val_labels)
    calibrated = apply_temperature(test_raw, temperature)
    return {
        "model": model,
        "served_temperature": served,
        "fitted_temperature": temperature,
        "held_out_ece_gain": held_out_improvement(val_raw, val_labels),
        "test": {
            "accuracy": float((test_raw.argmax(axis=1) == test_labels).mean()),
            "mean_confidence_before": float(test_raw.max(axis=1).mean()),
            "mean_confidence_after": float(calibrated.max(axis=1).mean()),
            "ece_before": expected_calibration_error(test_raw, test_labels),
            "ece_after": expected_calibration_error(calibrated, test_labels),
        },
        "bins_before": reliability_bins(test_raw, test_labels),
        "bins_after": reliability_bins(calibrated, test_labels),
    }


def run() -> list[dict]:
    names = {
        r["model"]: r.get("display_name", r["model"]) for r in evaluate.load_reports()
    }
    records = []
    for model in models_with_both_splits():
        record = compare_model(model)
        record["display_name"] = names.get(model, model)
        records.append(record)
        path = FIGURES_DIR / f"reliability_{model}.svg"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(reliability_svg(record), encoding="utf-8")
        LOGGER.info(
            "%s: T=%.2f, test ECE %.4f -> %.4f",
            model,
            record["fitted_temperature"],
            record["test"]["ece_before"],
            record["test"]["ece_after"],
        )
    if not records:
        LOGGER.warning("No validation and test predictions saved; run src.evaluate.")
        return []
    save_json(
        {"fitted_on": "validation", "measured_on": "test", "models": records},
        COMPARISON_PATH,
    )
    return records


# --- the table -----------------------------------------------------------------


def markdown_table(records: list[dict]) -> str:
    """Temperatures and test calibration, most accurate first."""
    rows = [
        "| Model | Fitted T | Accuracy | Mean confidence | ECE before | ECE after |",
        "|---|---|---|---|---|---|",
    ]
    for record in sorted(records, key=lambda r: -r["test"]["accuracy"]):
        test = record["test"]
        rows.append(
            f"| {record['display_name']} | {record['fitted_temperature']:.2f} "
            f"| {test['accuracy']:.4f} | {test['mean_confidence_before']:.3f} "
            f"| {test['ece_before']:.4f} | {test['ece_after']:.4f} |"
        )
    return "\n".join(rows)


# --- the diagram ---------------------------------------------------------------
#
# Hand-written SVG: one chart type, no plotting dependency in the locked
# requirements, and it renders on GitHub. Colours are the reference data-viz
# palette's first two categorical slots, validated as an adjacent pair in both
# light and dark modes; text uses ink tokens, never a series colour.

_WIDTH, _HEIGHT = 440, 400
_LEFT, _RIGHT, _TOP, _BOTTOM = 56, 24, 76, 52
_X_MIN = 0.3  # three classes: confidence can never fall below 1/3


def _x(value: float) -> float:
    return _LEFT + (value - _X_MIN) / (1 - _X_MIN) * (_WIDTH - _LEFT - _RIGHT)


def _y(value: float) -> float:
    return _HEIGHT - _BOTTOM - value * (_HEIGHT - _TOP - _BOTTOM)


_STYLE = """
  .surface { fill: #fcfcfb; }
  .ink { fill: #0b0b0b; font: 600 14px system-ui, sans-serif; }
  .ink-2 { fill: #52514e; font: 12px system-ui, sans-serif; }
  .grid { stroke: #e4e3df; stroke-width: 1; }
  .axis { stroke: #8a8984; stroke-width: 1; }
  .ideal { stroke: #8a8984; stroke-width: 1.5; stroke-dasharray: 4 4; fill: none; }
  .s1 { stroke: #2a78d6; fill: #2a78d6; }
  .s2 { stroke: #eb6834; fill: #eb6834; }
  .line { fill: none; stroke-width: 2; stroke-linejoin: round; }
  .dot { stroke: #fcfcfb; stroke-width: 2; }
  @media (prefers-color-scheme: dark) {
    .surface { fill: #1a1a19; }
    .ink { fill: #ffffff; }
    .ink-2 { fill: #c3c2b7; }
    .grid { stroke: #2e2e2c; }
    .s1 { stroke: #3987e5; fill: #3987e5; }
    .s2 { stroke: #d95926; fill: #d95926; }
    .dot { stroke: #1a1a19; }
  }
"""


def _series(bins: list[dict], css: str, name: str) -> str:
    points = " ".join(f"{_x(b['confidence']):.1f},{_y(b['accuracy']):.1f}" for b in bins)
    parts = [f'<polyline class="line {css}" points="{points}"/>']
    for b in bins:
        # Area-scaled markers would hide small bins; a fixed size with the
        # count in the tooltip keeps every bin readable.
        parts.append(
            f'<circle class="dot {css}" cx="{_x(b["confidence"]):.1f}" '
            f'cy="{_y(b["accuracy"]):.1f}" r="4.5"><title>{escape(name)}: '
            f"confidence {b['confidence']:.3f}, accuracy {b['accuracy']:.3f}, "
            f"{b['count']} tweets</title></circle>"
        )
    return "\n  ".join(parts)


def reliability_svg(record: dict) -> str:
    """A reliability diagram: accuracy against confidence, before and after."""
    test = record["test"]
    adopted = abs(record["fitted_temperature"] - 1.0) > 1e-3
    title = escape(record["display_name"])
    subtitle = (
        f"Test split. T = {record['fitted_temperature']:.2f} fitted on validation; "
        f"ECE {test['ece_before']:.3f} to {test['ece_after']:.3f}"
        if adopted
        else f"Test split. Fitted T = 1.00 (no softening); ECE {test['ece_before']:.3f}"
    )

    grid = []
    for tick in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        grid.append(
            f'<line class="grid" x1="{_LEFT}" x2="{_WIDTH - _RIGHT}" '
            f'y1="{_y(tick):.1f}" y2="{_y(tick):.1f}"/>'
            f'<text class="ink-2" x="{_LEFT - 8}" y="{_y(tick) + 4:.1f}" '
            f'text-anchor="end">{tick:.1f}</text>'
        )
    for tick in (0.4, 0.6, 0.8, 1.0):
        grid.append(
            f'<text class="ink-2" x="{_x(tick):.1f}" y="{_HEIGHT - _BOTTOM + 18}" '
            f'text-anchor="middle">{tick:.1f}</text>'
        )

    legend_y = _TOP - 18
    legend = [
        f'<circle class="dot s1" cx="{_LEFT + 6}" cy="{legend_y - 4}" r="4.5"/>'
        f'<text class="ink-2" x="{_LEFT + 16}" y="{legend_y}">Raw model output</text>'
    ]
    series = [_series(record["bins_before"], "s1", "Raw")]
    if adopted:
        legend.append(
            f'<circle class="dot s2" cx="{_LEFT + 156}" cy="{legend_y - 4}" r="4.5"/>'
            f'<text class="ink-2" x="{_LEFT + 166}" y="{legend_y}">'
            "After temperature scaling</text>"
        )
        series.append(_series(record["bins_after"], "s2", "Calibrated"))
    legend.append(
        f'<line class="ideal" x1="{_WIDTH - _RIGHT - 118}" x2="{_WIDTH - _RIGHT - 96}" '
        f'y1="{legend_y - 4}" y2="{legend_y - 4}"/>'
        f'<text class="ink-2" x="{_WIDTH - _RIGHT - 90}" y="{legend_y}">Perfect</text>'
    )

    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {_WIDTH} {_HEIGHT}" \
width="{_WIDTH}" height="{_HEIGHT}" role="img" aria-labelledby="t d">
<title id="t">Reliability diagram: {title}</title>
<desc id="d">{escape(subtitle)}. Points below the dashed diagonal are \
overconfident.</desc>
<style>{_STYLE}</style>
<rect class="surface" width="{_WIDTH}" height="{_HEIGHT}" rx="8"/>
<text class="ink" x="{_LEFT - 40}" y="24">{title}</text>
<text class="ink-2" x="{_LEFT - 40}" y="42">{escape(subtitle)}</text>
{"".join(legend)}
{"".join(grid)}
<line class="axis" x1="{_LEFT}" x2="{_WIDTH - _RIGHT}" y1="{_y(0):.1f}" y2="{_y(0):.1f}"/>
<line class="ideal" x1="{_x(_X_MIN):.1f}" y1="{_y(_X_MIN):.1f}" x2="{_x(1):.1f}" \
y2="{_y(1):.1f}"/>
<text class="ink-2" x="{(_LEFT + _WIDTH - _RIGHT) / 2:.1f}" y="{_HEIGHT - 12}" \
text-anchor="middle">Confidence (top-class probability)</text>
<text class="ink-2" transform="translate(16 {(_TOP + _HEIGHT - _BOTTOM) / 2:.1f}) \
rotate(-90)" text-anchor="middle">Accuracy</text>
  {(chr(10) + "  ").join(series)}
</svg>
"""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--table-only", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    records = load_json(COMPARISON_PATH)["models"] if args.table_only else run()
    if not records:
        return 1
    print()
    print(markdown_table(records))
    return 0


if __name__ == "__main__":
    sys.exit(main())
