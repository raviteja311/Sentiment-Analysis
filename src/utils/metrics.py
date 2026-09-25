from datetime import UTC, datetime

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)

from src.config import MODEL_DISPLAY_NAMES


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    acc = float(accuracy_score(y_true, y_pred))
    f1_macro = float(f1_score(y_true, y_pred, average="macro"))
    # TweetEval's official metric for the sentiment task, so scores here can
    # be set beside the benchmark's published numbers.
    recall_macro = float(recall_score(y_true, y_pred, average="macro"))
    f1_per_class = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2]).tolist()
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()
    cls_report = classification_report(y_true, y_pred, output_dict=True)
    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "recall_macro": recall_macro,
        "f1_per_class": f1_per_class,
        "confusion_matrix": cm,
        "classification_report": cls_report,
    }


def build_report(model, hyperparameters, dataset, metrics):
    """Assemble the JSON record written to reports/metrics/<model>.json.

    Scores on their own are not evidence: the same model scores differently on
    a different split, and the project's documentation once claimed 92-95%
    accuracy for a model whose own checkpoint recorded 78.9%. So a record
    carries the hyperparameters that produced it and the dataset it was measured
    on, alongside the numbers.
    """
    return {
        "model": model,
        "display_name": MODEL_DISPLAY_NAMES.get(model, model),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "hyperparameters": hyperparameters,
        "dataset": dataset,
        "metrics": metrics,
    }
