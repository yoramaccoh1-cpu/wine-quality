import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)


def evaluate_test(pipeline, X_test, y_test, threshold: float = 0.5) -> dict:
    """Compute F1 and ROC-AUC on the test set."""
    proba = pipeline.predict_proba(X_test)[:, 1]
    preds = (proba >= threshold).astype(int)
    return {
        "f1": round(f1_score(y_test, preds), 3),
        "roc_auc": round(roc_auc_score(y_test, proba), 3),
    }

def plot_confusion_matrix(pipeline, X_test, y_test, threshold: float = 0.5) -> None:
    """Plot the confusion matrix on the test set."""
    proba = pipeline.predict_proba(X_test)[:, 1]
    preds = (proba >= threshold).astype(int)
    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=["not good", "good"])
    disp.plot(cmap="Purples")
    plt.title(f"Confusion matrix (threshold = {threshold})")
    plt.tight_layout()
    plt.show()



def find_best_threshold(pipeline, X_val, y_val) -> float:
    """Find the threshold that maximises F1 on validation data."""
    proba = pipeline.predict_proba(X_val)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_val, proba)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-9)
    best_idx = int(np.argmax(f1_scores[:-1]))
    return float(thresholds[best_idx])

