import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    acc = float(accuracy_score(y_true, y_pred))
    f1_macro = float(f1_score(y_true, y_pred, average="macro"))
    f1_per_class = f1_score(y_true, y_pred, average=None, labels=[0,1,2]).tolist()
    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2]).tolist()
    cls_report = classification_report(y_true, y_pred, output_dict=True)
    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_per_class": f1_per_class,
        "confusion_matrix": cm,
        "classification_report": cls_report
    }
