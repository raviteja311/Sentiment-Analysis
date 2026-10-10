"""The Phase 2 grid: resumable runs, seed aggregation and the serving rule."""

import numpy as np
import pytest

from src import evaluate, finetune_grid
from src.config import BASELINE_KEY
from src.utils.io import save_json
from src.utils.metrics import compute_metrics

N = 600


@pytest.fixture
def scratch(tmp_path, monkeypatch):
    monkeypatch.setattr(finetune_grid, "GRID_DIR", tmp_path / "grid")
    monkeypatch.setattr(finetune_grid, "SMOKE_DIR", tmp_path / "smoke")
    monkeypatch.setattr(finetune_grid, "SUMMARY_PATH", tmp_path / "grid.json")
    monkeypatch.setattr(evaluate, "PREDICTIONS_DIR", tmp_path / "predictions")
    return tmp_path


def _labels():
    return np.random.default_rng(0).integers(0, 3, N)


def _predict(labels, accuracy, seed):
    rng = np.random.default_rng(seed)
    wrong = rng.random(len(labels)) > accuracy
    return np.where(wrong, (labels + 1) % 3, labels)


def _probs(predictions):
    probs = np.full((len(predictions), 3), 0.1)
    probs[np.arange(len(predictions)), predictions] = 0.8
    return probs


def _save_base(labels, accuracy):
    evaluate.save_predictions(
        BASELINE_KEY, "test", labels, _probs(_predict(labels, accuracy, seed=99))
    )


def _save_cell(run, seed, labels, accuracy, limit=None):
    directory = finetune_grid.cell_dir(run, seed)
    test = _predict(labels, accuracy, seed)
    dev_labels = labels[:200]
    dev = _predict(dev_labels, accuracy, seed + 1)
    evaluate.write_predictions(
        directory / "predictions" / "test.npz", labels, _probs(test)
    )
    evaluate.write_predictions(
        directory / "predictions" / "dev.npz", dev_labels, _probs(dev)
    )
    save_json(
        {
            "hyperparameters": {"seed": seed, "limit": limit},
            "metrics": {
                "test": compute_metrics(labels, test),
                "dev": compute_metrics(dev_labels, dev),
            },
        },
        directory / "metrics.json",
    )


def _fill_grid(labels, accuracies):
    for run, accuracy in accuracies.items():
        for seed in finetune_grid.SEEDS:
            _save_cell(run, seed, labels, accuracy)


# --- running ---------------------------------------------------------------------


def test_finished_cells_are_skipped(scratch, monkeypatch):
    calls = []
    monkeypatch.setattr(
        finetune_grid, "train_cell", lambda run, seed, **_: calls.append((run, seed))
    )
    _save_cell("B", 13, _labels(), 0.7)

    finetune_grid.run_grid(["B"], seeds=[13, 42])

    assert calls == [("B", 42)]


def test_smoke_tests_never_land_in_the_grid(scratch, monkeypatch):
    from src.training import train_roberta

    seen = {}
    monkeypatch.setattr(
        train_roberta, "main", lambda config, out_dir, options: seen.update(out=out_dir)
    )
    finetune_grid.train_cell("C", 13, limit=64)
    assert seen["out"].parent == finetune_grid.SMOKE_DIR


def test_grid_cells_use_the_preregistered_settings(scratch, monkeypatch):
    from src.training import train_roberta

    seen = {}
    monkeypatch.setattr(
        train_roberta,
        "main",
        lambda config, out_dir, options: seen.update(config=config, options=options),
    )
    finetune_grid.train_cell("C", 42)

    config, options = seen["config"], seen["options"]
    assert config.base_model == "cardiffnlp/twitter-roberta-base"
    assert (config.learning_rate, config.batch_size, config.epochs) == (2e-5, 32, 3)
    assert config.warmup_ratio == 0.06
    assert options.select_on == "dev" and options.select_metric == "recall_macro"
    assert options.preprocessing == "cardiff-v1" and options.seed == 42
    assert not options.save_weights and options.save_predictions


# --- the rule ----------------------------------------------------------------------


def test_no_decision_until_every_run_has_every_seed(scratch):
    labels = _labels()
    _save_base(labels, 0.7)
    _fill_grid(labels, {"A": 0.6, "B": 0.9, "C": 0.7})  # D missing

    result = finetune_grid.summarise()

    assert result["decision"]["decision"] == "incomplete"


def test_a_clearly_better_run_is_served(scratch):
    labels = _labels()
    _save_base(labels, 0.7)
    _fill_grid(labels, {"A": 0.6, "B": 0.9, "C": 0.7, "D": 0.65})

    result = finetune_grid.summarise()

    decision = result["decision"]
    assert decision["winner"] == "B"
    assert decision["gain_over_base"]["interval"][0] > 0
    assert decision["decision"].startswith("serve B seed ")
    assert "| B |" in finetune_grid.markdown_table(result)


def test_a_gain_within_noise_serves_the_base(scratch):
    labels = _labels()
    _save_base(labels, 0.70)
    _fill_grid(labels, {"A": 0.60, "B": 0.705, "C": 0.69, "D": 0.65})

    decision = finetune_grid.summarise()["decision"]

    assert decision["decision"] == "serve base checkpoint"


def test_seed_spread_is_reported(scratch):
    labels = _labels()
    _save_base(labels, 0.7)
    _fill_grid(labels, {"A": 0.6, "B": 0.8, "C": 0.7, "D": 0.65})

    run = next(r for r in finetune_grid.summarise()["runs"] if r["run"] == "B")

    assert run["seeds"] == list(finetune_grid.SEEDS)
    assert run["test"]["recall_macro"]["std"] > 0
    low, high = run["test_interval"]["recall_macro"]
    assert low < run["test"]["recall_macro"]["mean"] < high
    assert run["best_dev_seed"] in finetune_grid.SEEDS
