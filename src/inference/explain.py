"""What drove a linear prediction: the terms and their weights.

For the TF-IDF + LogisticRegression pipeline a prediction is a sum, term by
term, of tf-idf value times class coefficient, so it can be shown exactly.
Nothing here applies to the neural models, whose decisions are not sums of
per-term weights; the UI only offers this for the linear model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TermContribution:
    term: str
    # Coefficient of the term for the class in question.
    weight: float
    # The term's tf-idf value in this text.
    tfidf: float
    # weight * tfidf: what the term added to the class score. Positive pushes
    # towards the class, negative away from it.
    contribution: float


def top_weighted_terms(pipeline, cleaned_text: str, class_index: int, k: int = 10):
    """The ``k`` terms that moved this text's score for ``class_index`` most.

    ``cleaned_text`` must already be preprocessed with the spec the pipeline
    was trained on; the predictor does that. Terms absent from the text or
    from the vocabulary contribute nothing and are not listed. Ordered by the
    size of the contribution, sign aside, so the strongest push either way
    comes first.
    """
    tfidf = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["clf"]
    # predict_proba columns follow classes_, not label index; so do the rows
    # of coef_. Look the class up rather than assuming they coincide.
    column = int(np.flatnonzero(classifier.classes_ == class_index)[0])

    row = tfidf.transform([cleaned_text])
    if row.nnz == 0:
        return []
    row = row.tocoo()
    names = tfidf.get_feature_names_out()
    weights = classifier.coef_[column]

    terms = [
        TermContribution(
            term=str(names[index]),
            weight=float(weights[index]),
            tfidf=float(value),
            contribution=float(weights[index] * value),
        )
        for index, value in zip(row.col, row.data, strict=True)
    ]
    terms.sort(key=lambda t: -abs(t.contribution))
    return terms[:k]
