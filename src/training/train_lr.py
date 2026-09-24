"""Train the TF-IDF + LogisticRegression baseline.

Every setting the vectorizer and the classifier use comes from
``config.LR_CONFIG``, and the same dataclass is what gets written into the
metrics record as the run's hyperparameters. The two must stay in lock-step:
a field that exists in the config but is not passed to the estimator produces a
record that describes a model which was never trained.
"""

import logging
from dataclasses import asdict

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from src.config import LR_CONFIG, LR_DIR, METRICS_DIR, SEED
from src.data import describe, load_raw_dataset
from src.utils.io import save_json
from src.utils.metrics import build_report, compute_metrics
from src.utils.preprocessing import preprocess_tweet

OUT_DIR = LR_DIR


def build_pipeline(config=LR_CONFIG) -> Pipeline:
    """TF-IDF + LogisticRegression pipeline, built from the shared config.

    Kept separate from :func:`main` so the pipeline can be constructed - and
    therefore checked - without downloading the dataset.
    """
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=config.ngram_range,
                    min_df=config.min_df,
                    max_df=config.max_df,
                    max_features=config.max_features,
                    # Without this the vectorizer fell back to scikit-learn's
                    # default pattern, which drops the angle brackets and folds
                    # `<user>` into the ordinary word "user" - while the metrics
                    # record, built from the same config, claimed otherwise.
                    token_pattern=config.token_pattern,
                ),
            ),
            # `multi_class='multinomial'` used to be passed here. scikit-learn 1.7
            # removed the argument, so every install newer than that raised
            # TypeError before training could start. Multinomial is the default for
            # the saga solver anyway, so the behaviour is unchanged.
            (
                "clf",
                LogisticRegression(
                    C=config.C,
                    max_iter=config.max_iter,
                    class_weight=config.class_weight,
                    solver=config.solver,
                    # saga shuffles the data each epoch. The model card says
                    # training is seeded; without this the linear baseline was
                    # the one model for which that was not true.
                    random_state=SEED,
                ),
            ),
        ]
    )


def ds_to_df(ds, split):
    import pandas as pd

    texts = [preprocess_tweet(t) for t in ds[split]["text"]]
    labels = ds[split]["label"]
    return pd.DataFrame({"text": texts, "label": labels})


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # The shared loader, so the dataset name comes from config and the split
    # sizes are logged like every other training run's.
    ds = load_raw_dataset()
    train_df = ds_to_df(ds, "train")
    val_df = ds_to_df(ds, "validation")
    test_df = ds_to_df(ds, "test")

    pipe = build_pipeline()

    params = {"tfidf__max_features": [LR_CONFIG.max_features], "clf__C": [LR_CONFIG.C]}

    print("Starting GridSearchCV (this may take a while)...")
    # n_jobs here is GridSearchCV's, which is still supported; it is
    # LogisticRegression's own n_jobs that scikit-learn deprecated.
    gs = GridSearchCV(pipe, params, cv=3, scoring="f1_macro", n_jobs=-1, verbose=1)
    gs.fit(train_df["text"], train_df["label"])

    best = gs.best_estimator_
    print("Best params:", gs.best_params_)

    metrics = {
        "validation": compute_metrics(val_df["label"], best.predict(val_df["text"])),
        "test": compute_metrics(test_df["label"], best.predict(test_df["text"])),
    }
    for split, scores in metrics.items():
        print(
            f"{split}: accuracy={scores['accuracy']:.4f} "
            f"macro F1={scores['f1_macro']:.4f}"
        )

    model_path = OUT_DIR / "pipeline.joblib"
    joblib.dump(best, model_path)
    print(f"Saved LR pipeline to: {model_path}")

    # Metrics are written, not just printed. Printing them and throwing them
    # away is how the documented numbers ended up unverifiable.
    report = build_report(
        model="lr",
        hyperparameters=asdict(LR_CONFIG),
        dataset=describe(ds),
        metrics=metrics,
    )
    report_path = METRICS_DIR / "lr.json"
    save_json(report, report_path)
    print(f"Wrote metrics: {report_path}")

    return report


if __name__ == "__main__":
    main()
