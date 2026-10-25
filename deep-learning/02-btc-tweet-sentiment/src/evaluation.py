"""Shared evaluation helpers for binary sentiment classifiers."""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

METRIC_NAMES = ["Accuracy", "Precision", "Recall", "F1", "AUC"]


def evaluate_binary_classifier(y_true, y_score, cutoff=0.5, model_name=None, verbose=True):
    """
    Evaluate a binary classifier from continuous scores.

    Parameters
    ----------
    y_true : array-like of 0/1 labels
    y_score : array-like of scores; higher means more likely positive
        (probabilities, or e.g. dictionary polarity in [-1, 1])
    cutoff : scores above this value are classified as positive

    Returns
    -------
    dict with keys Accuracy, Precision, Recall, F1, AUC and ConfusionMatrix.
    Returning a dict (instead of a tuple) means results are always looked up
    by name, so metrics cannot end up in the wrong row of the results table.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    y_pred = (y_score > cutoff).astype(int)

    results = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_score),
        "ConfusionMatrix": confusion_matrix(y_true, y_pred),
    }

    if verbose:
        label = f"{model_name}: " if model_name else ""
        print(label + "  ".join(f"{m}={results[m]:.3f}" for m in METRIC_NAMES))
        print("Confusion matrix [[TN FP] [FN TP]]:")
        print(results["ConfusionMatrix"])
    return results


def plot_roc_curves(y_true, scores_by_model, ax=None):
    """Plot ROC curves of several models in one chart."""
    ax = ax or plt.subplots(figsize=(6, 5))[1]
    for name, y_score in scores_by_model.items():
        fpr, tpr, _ = roc_curve(y_true, y_score)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_true, y_score):.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves on the test set")
    ax.legend(loc="lower right", fontsize=8)
    return ax


def plot_training_history(history, title="Training vs. validation loss", ax=None):
    """Plot train and validation loss from a Keras History object."""
    ax = ax or plt.subplots(figsize=(6, 4))[1]
    ax.plot(history.history["loss"], label="train")
    ax.plot(history.history["val_loss"], label="validation")
    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.set_ylabel("binary cross-entropy")
    ax.legend()
    return ax
