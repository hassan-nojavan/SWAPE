import numpy as np

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


def standardize_train_test(X_train, X_test):
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, scaler


def standardize_with_scaler(X, scaler):
    return scaler.transform(X)


def classification_metrics(y_true, probs):
    predictions = (probs >= 0.5).astype(int)

    accuracy = accuracy_score(
        y_true,
        predictions,
    )

    auc = roc_auc_score(
        y_true,
        probs,
    )

    return {
        "accuracy": float(accuracy),
        "auc": float(auc),
    }


def weight_metrics(
    w,
    contaminated,
    atol=1e-8,
):
    w = np.asarray(w, dtype=float)
    contaminated = np.asarray(
        contaminated,
        dtype=bool,
    )

    # Zero weights are treated as detected flipped labels
    detected = w <= atol

    tp = np.sum(detected & contaminated)
    fp = np.sum(detected & ~contaminated)
    fn = np.sum(~detected & contaminated)

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0.0
    )

    f1 = (
        2.0 * precision * recall
        / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    retained = 100.0 * np.mean(~detected)

    return {
        "retained": float(retained),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def evaluate_neural_network(
    y_test,
    probs,
    w,
    contaminated,
):
    metrics = classification_metrics(
        y_test,
        probs,
    )

    metrics.update(
        weight_metrics(
            w,
            contaminated,
        )
    )

    return metrics