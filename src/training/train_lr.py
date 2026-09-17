# src/training/train_lr.py
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from src.config import LR_CONFIG, LR_DIR
from src.utils.preprocessing import preprocess_tweet
from src.utils.metrics import compute_metrics

OUT_DIR = LR_DIR


def build_pipeline(config=LR_CONFIG) -> Pipeline:
    """TF-IDF + LogisticRegression pipeline, built from the shared config.

    Kept separate from :func:`main` so the pipeline can be constructed - and
    therefore checked - without downloading the dataset.
    """
    return Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=config.ngram_range,
            min_df=config.min_df,
            max_df=config.max_df,
            max_features=config.max_features,
        )),
        # `multi_class='multinomial'` used to be passed here. scikit-learn 1.7
        # removed the argument, so every install newer than that raised
        # TypeError before training could start. Multinomial is the default for
        # the saga solver anyway, so the behaviour is unchanged.
        ('clf', LogisticRegression(
            C=config.C,
            max_iter=config.max_iter,
            class_weight=config.class_weight,
            solver=config.solver,
        )),
    ])


def ds_to_df(ds, split):
    import pandas as pd

    texts = [preprocess_tweet(t) for t in ds[split]['text']]
    labels = ds[split]['label']
    return pd.DataFrame({"text": texts, "label": labels})


def main():
    from datasets import load_dataset

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    ds = load_dataset("cardiffnlp/tweet_eval", "sentiment")
    train_df = ds_to_df(ds, "train")
    val_df = ds_to_df(ds, "validation")
    test_df = ds_to_df(ds, "test")

    pipe = build_pipeline()

    params = {
        'tfidf__max_features': [LR_CONFIG.max_features],
        'clf__C': [LR_CONFIG.C]
    }

    print("Starting GridSearchCV (this may take a while)...")
    # n_jobs here is GridSearchCV's, which is still supported; it is
    # LogisticRegression's own n_jobs that scikit-learn deprecated.
    gs = GridSearchCV(pipe, params, cv=3, scoring='f1_macro', n_jobs=-1, verbose=1)
    gs.fit(train_df['text'], train_df['label'])

    best = gs.best_estimator_
    print("Best params:", gs.best_params_)

    val_preds = best.predict(val_df['text'])
    test_preds = best.predict(test_df['text'])

    print("Validation:", compute_metrics(val_df['label'], val_preds))
    print("Test:", compute_metrics(test_df['label'], test_preds))

    model_path = OUT_DIR / "pipeline.joblib"
    joblib.dump(best, model_path)
    print(f"Saved LR pipeline to: {model_path}")


if __name__ == "__main__":
    main()
