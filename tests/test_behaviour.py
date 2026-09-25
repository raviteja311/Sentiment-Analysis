"""The behavioural check harness, tested with stub labellers.

What the models actually do is a report, not a test; what the harness does
with their answers is.
"""

import pytest

from src import behaviour
from src.behaviour import FLIP, SAME, Case, run_check, run_suite


def always(label):
    return lambda texts: [label] * len(texts)


def by_keyword(texts):
    """A labeller that only understands a few words, like a tiny model."""
    out = []
    for text in texts:
        lowered = text.lower()
        if "not" in lowered or "n't" in lowered or "awful" in lowered:
            out.append("negative")
        elif "like" in lowered or "good" in lowered or "great" in lowered:
            out.append("positive")
        else:
            out.append("neutral")
    return out


def test_a_flip_case_passes_when_the_labels_differ():
    result = run_check(by_keyword, [Case("I like it", FLIP, "I don't like it")])
    assert result == {"passed": 1, "total": 1, "pass_rate": 1.0, "failures": []}


def test_a_flip_case_fails_when_the_labels_agree():
    result = run_check(always("positive"), [Case("I like it", FLIP, "I don't like it")])
    assert result["passed"] == 0
    assert result["failures"] == [
        {
            "text": "I like it",
            "other": "I don't like it",
            "labels": ["positive", "positive"],
        }
    ]


def test_a_same_case_passes_when_the_labels_agree():
    cases = [Case("@a it is awful", SAME, "@b it is awful")]
    assert run_check(by_keyword, cases)["passed"] == 1


def test_a_same_case_fails_when_the_labels_differ():
    def flaky(texts):
        return ["positive", "negative"]

    result = run_check(flaky, [Case("@a hi", SAME, "@b hi")])
    assert result["passed"] == 0
    assert result["failures"][0]["labels"] == ["positive", "negative"]


def test_an_expected_label_case_compares_against_the_label():
    cases = [Case("<3 you", "positive"), Case("ugh", "negative")]
    result = run_check(always("positive"), cases)
    assert result["passed"] == 1
    assert result["failures"] == [
        {"text": "ugh", "expected": "negative", "label": "positive"}
    ]


def test_the_labeller_is_called_once_per_check():
    # One batched call, so a transformer scores a check in one forward pass.
    calls = []

    def counting(texts):
        calls.append(list(texts))
        return ["neutral"] * len(texts)

    run_check(
        counting, [Case("a", FLIP, "b"), Case("c", "neutral"), Case("d", SAME, "e")]
    )
    assert calls == [["a", "c", "d", "b", "e"]]


def test_the_suite_aggregates_across_checks():
    result = run_suite(always("positive"))
    assert set(result["checks"]) == set(behaviour.SUITE)
    assert result["total"] == sum(len(cases) for cases in behaviour.SUITE.values())
    # A model that says "positive" to everything passes every invariance
    # check, every positive expectation, and no negation check.
    assert result["checks"]["negation flips the label"]["passed"] == 0
    assert result["checks"]["mention invariance"]["pass_rate"] == 1.0
    assert result["checks"]["url invariance"]["pass_rate"] == 1.0
    assert 0 < result["pass_rate"] < 1


def test_every_case_is_well_formed():
    for cases in behaviour.SUITE.values():
        for case in cases:
            if case.expect in (FLIP, SAME):
                assert case.other is not None and case.other != case.text
            else:
                assert case.expect in ("negative", "neutral", "positive")
                assert case.other is None


def test_check_model_writes_a_report(monkeypatch, tmp_path):
    class Stub:
        version = "abc123"

        def predict_many(self, texts):
            return [type("P", (), {"label": label})() for label in by_keyword(texts)]

    monkeypatch.setattr(behaviour, "METRICS_DIR", tmp_path)
    monkeypatch.setattr("src.inference.predictor.load_predictor", lambda model: Stub())

    result = behaviour.check_model("bert")  # alias resolves like everywhere else

    assert result["model"] == "roberta"
    assert result["version"] == "abc123"
    assert (tmp_path / "behaviour_roberta.json").is_file()
    assert result["checks"]["negation flips the label"]["pass_rate"] == 1.0


def test_the_table_has_one_column_per_model():
    results = [
        dict(run_suite(always("positive")), model="lr"),
        dict(run_suite(by_keyword), model="gru"),
    ]
    table = behaviour.markdown_table(results)
    assert table.splitlines()[0] == "| Check | lr | gru |"
    assert "negation flips the label" in table
    assert "**all**" in table


def test_the_table_says_so_when_nothing_was_checked():
    assert "No models checked" in behaviour.markdown_table([])


def test_failing_checks_do_not_fail_the_run(monkeypatch, tmp_path, capsys):
    # A report, not a gate.
    class Stub:
        version = None

        def predict_many(self, texts):
            return [type("P", (), {"label": "positive"})() for _ in texts]

    monkeypatch.setattr(behaviour, "METRICS_DIR", tmp_path)
    monkeypatch.setattr("src.inference.predictor.load_predictor", lambda model: Stub())
    monkeypatch.setattr("src.inference.predictor.available_models", lambda: ["lr"])

    assert behaviour.main([]) == 0
    assert "| lr |" in capsys.readouterr().out


def test_an_unavailable_model_is_skipped(monkeypatch, tmp_path):
    from src.inference.predictor import ModelUnavailableError

    def unavailable(model):
        raise ModelUnavailableError("weights missing")

    monkeypatch.setattr(behaviour, "METRICS_DIR", tmp_path)
    monkeypatch.setattr("src.inference.predictor.load_predictor", unavailable)
    assert behaviour.main(["--models", "gru"]) == 0
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("model", ["lr"])
def test_the_real_lr_model_can_be_checked(model, monkeypatch, tmp_path):
    from src.inference.predictor import check_artifacts

    if check_artifacts(model) is not None:
        pytest.skip("lr artifacts unavailable")
    monkeypatch.setattr(behaviour, "METRICS_DIR", tmp_path)
    result = behaviour.check_model(model)
    assert result["total"] > 0
    assert 0.0 <= result["pass_rate"] <= 1.0
