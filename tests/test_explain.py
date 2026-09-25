"""Term contributions for the linear model."""

import numpy as np
import pytest

from src.inference.explain import top_weighted_terms

pytest.importorskip("sklearn")


@pytest.fixture(scope="module")
def tiny_pipeline():
    """A pipeline fitted on a handful of documents, so no artifact is needed."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    docs = [
        ("awful terrible hate", 0),
        ("awful bad worst", 0),
        ("meeting schedule update", 1),
        ("agenda meeting notes", 1),
        ("great love wonderful", 2),
        ("great happy best", 2),
    ] * 3
    pipeline = Pipeline(
        [("tfidf", TfidfVectorizer()), ("clf", LogisticRegression(max_iter=500))]
    )
    pipeline.fit([d for d, _ in docs], [label for _, label in docs])
    return pipeline


def test_terms_come_from_the_text_and_the_vocabulary(tiny_pipeline):
    terms = top_weighted_terms(tiny_pipeline, "awful meeting unknownword", 0)
    assert {t.term for t in terms} == {"awful", "meeting"}


def test_the_strongest_contribution_comes_first(tiny_pipeline):
    terms = top_weighted_terms(tiny_pipeline, "awful meeting great", 0)
    magnitudes = [abs(t.contribution) for t in terms]
    assert magnitudes == sorted(magnitudes, reverse=True)
    assert terms[0].term == "awful"
    assert terms[0].contribution > 0


def test_contribution_is_weight_times_tfidf(tiny_pipeline):
    for term in top_weighted_terms(tiny_pipeline, "awful meeting", 2):
        assert term.contribution == pytest.approx(term.weight * term.tfidf)


def test_a_word_that_argues_against_the_class_shows_as_negative(tiny_pipeline):
    (great,) = [
        t for t in top_weighted_terms(tiny_pipeline, "great", 0) if t.term == "great"
    ]
    assert great.contribution < 0


def test_k_bounds_the_list(tiny_pipeline):
    assert (
        len(top_weighted_terms(tiny_pipeline, "awful meeting great hate love", 2, k=2))
        == 2
    )


def test_text_with_nothing_in_the_vocabulary_gives_no_terms(tiny_pipeline):
    assert top_weighted_terms(tiny_pipeline, "zzz qqq", 1) == []


def test_the_class_is_looked_up_in_classes_not_assumed(tiny_pipeline):
    # Reorder classes_ and coef_ together; the terms for label 0 must not change.
    before = top_weighted_terms(tiny_pipeline, "awful meeting", 0)
    clf = tiny_pipeline.named_steps["clf"]
    order = [2, 0, 1]
    clf.classes_ = clf.classes_[order]
    clf.coef_ = clf.coef_[order]
    try:
        after = top_weighted_terms(tiny_pipeline, "awful meeting", 0)
    finally:
        inverse = np.argsort(order)
        clf.classes_ = clf.classes_[inverse]
        clf.coef_ = clf.coef_[inverse]
    assert [(t.term, t.contribution) for t in after] == [
        (t.term, t.contribution) for t in before
    ]
