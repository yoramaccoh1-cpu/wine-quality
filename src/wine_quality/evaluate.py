import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (f1_score,roc_auc_score,confusion_matrix,ConfusionMatrixDisplay,precision_recall_curve)


def evaluate_test(pipeline, X_test, y_test, threshold=0.5):
    """Calcule F1 et ROC-AUC sur le test set."""
    proba = pipeline.predict_proba(X_test)[:, 1]
    preds = (proba >= threshold).astype(int)
    return {
        "f1": round(f1_score(y_test, preds), 3),
        "roc_auc": round(roc_auc_score(y_test, proba), 3),
    }


def plot_confusion_matrix(pipeline, X_test, y_test, threshold=0.5):
    """Affiche la matrice de confusion sur le test set."""
    proba = pipeline.predict_proba(X_test)[:, 1]
    preds = (proba >= threshold).astype(int)
    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=["not good", "good"])
    disp.plot(cmap="Purples")
    plt.title(f"Confusion matrix (threshold = {threshold})")
    plt.tight_layout()
    plt.show()



def find_best_threshold(pipeline, X_val, y_val):
    """Trouve le seuil qui maximise le F1 sur les données de validation."""
    proba = pipeline.predict_proba(X_val)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_val, proba)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-9)
    best_idx = int(np.argmax(f1_scores[:-1]))
    return float(thresholds[best_idx])


if __name__ == "__main__":
    from wine_quality.data import DatasetLoader
    from wine_quality.models import Trainer

    loader = DatasetLoader()
    loader.load()
    X_train, X_test, y_train, y_test = loader.split()

    trainer = Trainer()
    trainer.compare(X_train, y_train)
    trainer.tune(X_train, y_train)
    trainer.save()

    metrics = evaluate_test(trainer.best_pipeline, X_test, y_test)
    print(metrics)
    plot_confusion_matrix(trainer.best_pipeline, X_test, y_test)