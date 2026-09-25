"""The validation-only experiment harness, with synthetic predictions."""

import numpy as np
import pytest

from src import experiments
from src.config import LABELS


def synthetic(seed=0, n=600, prior=(0.2, 0.5, 0.3), sharpness=3.0):
    """Labels drawn from ``prior`` and plausible, imperfect probabilities."""
    rng = np.random.default_rng(seed)
    labels = rng.choice(3, size=n, p=prior)
    logits = rng.normal(size=(n, 3))
    logits[np.arange(n), labels] += sharpness
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    return probs, labels


# --- decision bias ----------------------------------------------------------


def test_the_bias_favours_a_class_the_model_under_predicts():
    probs, labels = synthetic()
    # Suppress the positive class so it is systematically under-predicted.
    skewed = probs.copy()
    skewed[:, 2] *= 0.2
    skewed /= skewed.sum(axis=1, keepdims=True)
    bias = experiments.fit_decision_bias(skewed, labels)
    assert bias[0] == 0.0  # pinned by convention
    assert bias[2] > 0.0
    before = experiments.scores(labels, skewed.argmax(axis=1))["f1_macro"]
    after = experiments.scores(labels, experiments.apply_decision_bias(skewed, bias))[
        "f1_macro"
    ]
    assert after >= before


def test_the_bias_experiment_never_scores_the_fold_it_fitted():
    probs, labels = synthetic()
    result = experiments.decision_bias_experiment(probs, labels, seed=1)
    assert result["judged_on"].startswith("validation")
    assert len(result["folds"]) == 2
    # Each fold's bias is fitted on the other half, so the two are allowed to
    # differ; what must hold is that both halves were scored.
    assert all(fold["before"]["accuracy"] <= 1.0 for fold in result["folds"])
    assert set(result["delta"]) == {
        "accuracy",
        "f1_macro",
        "recall_macro",
        "f1_per_class",
    }


def test_the_bias_experiment_is_deterministic():
    probs, labels = synthetic()
    first = experiments.decision_bias_experiment(probs, labels)
    second = experiments.decision_bias_experiment(probs, labels)
    assert first["folds"][0]["bias"] == second["folds"][0]["bias"]


# --- label shift ----------------------------------------------------------------


def test_em_moves_the_prior_estimate_towards_the_truth():
    train_prior = np.array([0.155, 0.453, 0.391])
    true_prior = (0.5, 0.3, 0.2)
    # Posteriors from a model calibrated to the training prior, applied to
    # data drawn from a different one.
    probs, labels = synthetic(seed=3, n=4000, prior=true_prior, sharpness=2.0)
    shifted = probs * (train_prior / np.array(true_prior))
    shifted /= shifted.sum(axis=1, keepdims=True)

    estimated = experiments.estimate_priors_em(shifted, train_prior)
    assert abs(estimated.sum() - 1.0) < 1e-6
    for cls in range(3):
        assert abs(estimated[cls] - true_prior[cls]) < abs(
            train_prior[cls] - true_prior[cls]
        )


def test_applying_the_training_prior_changes_nothing():
    probs, _ = synthetic()
    prior = np.array([0.2, 0.5, 0.3])
    np.testing.assert_allclose(experiments.apply_priors(probs, prior, prior), probs)


def test_the_label_shift_experiment_records_every_prior():
    probs, labels = synthetic()
    result = experiments.label_shift_experiment(probs, labels, [0.2, 0.5, 0.3])
    assert set(result["train_priors"]) == set(LABELS)
    assert set(result["estimated_priors"]) == set(LABELS)
    assert abs(sum(result["actual_priors"].values()) - 1.0) < 1e-9
    assert result["judged_on"] == "validation"


# --- class weight ---------------------------------------------------------------


def test_the_class_weight_experiment_trains_into_scratch_and_compares_validation(
    monkeypatch, tmp_path
):
    from src.utils.io import save_json

    served = {
        "hyperparameters": {"class_weight": None},
        "metrics": {
            "validation": {
                "accuracy": 0.68,
                "f1_macro": 0.65,
                "recall_macro": 0.64,
                "f1_per_class": [0.56, 0.70, 0.70],
            }
        },
    }
    save_json(served, tmp_path / "metrics" / "gru.json")
    monkeypatch.setattr(experiments, "METRICS_DIR", tmp_path / "metrics")
    monkeypatch.setattr(experiments, "SCRATCH_DIR", tmp_path / "scratch")

    seen = {}

    def fake_train(key, build_fn, config, out_dir):
        seen.update(key=key, class_weight=config.class_weight, out_dir=out_dir)
        return {
            "metrics": {
                "validation": {
                    "accuracy": 0.66,
                    "f1_macro": 0.64,
                    "recall_macro": 0.66,
                    "f1_per_class": [0.60, 0.62, 0.70],
                }
            }
        }

    monkeypatch.setattr("src.training.train_sequence.train_sequence_model", fake_train)

    result = experiments.class_weight_experiment("gru")

    assert seen["key"] == "gru"
    assert seen["class_weight"] == "balanced"
    assert seen["out_dir"] == tmp_path / "scratch" / "class_weight_gru"
    assert result["delta"]["f1_macro"] == pytest.approx(-0.01)
    assert result["delta"]["f1_per_class"][0] == pytest.approx(0.04)


# --- running and reporting ----------------------------------------------------------


def test_run_writes_one_record_per_experiment_and_never_reads_test(monkeypatch, tmp_path):
    from src.utils.io import save_json

    probs, labels = synthetic()
    touched = []

    class Split(dict):
        def __getitem__(self, key):
            touched.append(key)
            return super().__getitem__(key)

    dataset = Split(validation={"text": ["t"] * len(labels), "label": labels})
    monkeypatch.setattr("src.data.load_raw_dataset", lambda: dataset)
    monkeypatch.setattr(
        experiments,
        "validation_probabilities",
        lambda model, ds, batch_size: (probs, labels),
    )
    monkeypatch.setattr(experiments, "EXPERIMENTS_DIR", tmp_path)
    save_json(
        {
            "dataset": {
                "class_distribution": {
                    "train": {"negative": 2, "neutral": 5, "positive": 3}
                }
            }
        },
        tmp_path / "metrics" / "lr.json",
    )
    monkeypatch.setattr(experiments, "METRICS_DIR", tmp_path / "metrics")

    results = experiments.run(["lr"], ("decision_bias", "label_shift"))

    assert [r["experiment"] for r in results] == ["decision_bias", "label_shift"]
    assert (tmp_path / "decision_bias_lr.json").is_file()
    assert (tmp_path / "label_shift_lr.json").is_file()
    assert "test" not in touched


def test_class_weight_is_skipped_for_models_that_are_not_sequence_models(monkeypatch):
    probs, labels = synthetic()
    monkeypatch.setattr("src.data.load_raw_dataset", lambda: {})
    monkeypatch.setattr(
        experiments,
        "validation_probabilities",
        lambda model, ds, batch_size: (probs, labels),
    )
    assert experiments.run(["lr"], ("class_weight",)) == []


def test_the_table_reads_before_to_after():
    probs, labels = synthetic()
    result = dict(experiments.decision_bias_experiment(probs, labels), model="gru")
    table = experiments.markdown_table([result])
    assert table.splitlines()[0].startswith(
        "| Model | Experiment | Accuracy | Macro F1 |"
    )
    assert "| gru | decision_bias |" in table
    assert " to " in table and "(" in table


def test_the_table_says_so_when_nothing_was_run():
    assert "No experiments recorded" in experiments.markdown_table([])


def test_an_unknown_experiment_is_rejected(monkeypatch):
    probs, labels = synthetic()
    monkeypatch.setattr("src.data.load_raw_dataset", lambda: {})
    monkeypatch.setattr(
        experiments,
        "validation_probabilities",
        lambda model, ds, batch_size: (probs, labels),
    )
    with pytest.raises(ValueError, match="Unknown experiment"):
        experiments.run(["lr"], ("nope",))
