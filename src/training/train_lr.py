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
from sklearn.pipeline import Pipeline

from src.config import LR_CONFIG, LR_DIR, METRICS_DIR, SEED, TRAIN_PREPROCESSING
from src.data import describe, load_raw_dataset, prepare_split
from src.utils.io import save_json
from src.utils.metrics import build_report, compute_metrics
from src.utils.preprocessing import write_artifact_spec

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


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # The shared loader and split preparation, so the dataset name comes from
    # config, the split sizes are logged like every other training run's, and
    # the text is preprocessed exactly as the other models see it.
    spec = TRAIN_PREPROCESSING["lr"]
    ds = load_raw_dataset()
    train_texts, train_labels = prepare_split(ds, "train", spec)
    val_texts, val_labels = prepare_split(ds, "validation", spec)
    test_texts, test_labels = prepare_split(ds, "test", spec)

    # Fitted directly. This used to go through GridSearchCV over a grid with
    # exactly one point, which fitted the same pipeline three times for cross-
    # validation and a fourth time on the full split - tripling the run for a
    # search that could only ever return the configured values.
    print("Fitting the TF-IDF + LogisticRegression pipeline...")
    best = build_pipeline()
    best.fit(train_texts, train_labels)

    metrics = {
        "validation": compute_metrics(val_labels, best.predict(val_texts)),
        "test": compute_metrics(test_labels, best.predict(test_texts)),
    }
    for split, scores in metrics.items():
        print(
            f"{split}: accuracy={scores['accuracy']:.4f} "
            f"macro F1={scores['f1_macro']:.4f}"
        )

    model_path = OUT_DIR / "pipeline.joblib"
    joblib.dump(best, model_path)
    print(f"Saved LR pipeline to: {model_path}")
    # The vocabulary only matches text normalised this way; inference reads
    # the spec back from beside the pipeline.
    write_artifact_spec("lr", spec)

    # Metrics are written, not just printed. Printing them and throwing them
    # away is how the documented numbers ended up unverifiable.
    report = build_report(
        model="lr",
        hyperparameters={**asdict(LR_CONFIG), "preprocessing": spec},
        dataset=describe(ds),
        metrics=metrics,
    )
    report_path = METRICS_DIR / "lr.json"
    save_json(report, report_path)
    print(f"Wrote metrics: {report_path}")

    return report


if __name__ == "__main__":
    main()
