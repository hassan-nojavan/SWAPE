import numpy as np


def prediction_metrics(A_test, b_test, x_hat, x_true):
    prediction = A_test @ x_hat

    mse = np.mean((prediction - b_test) ** 2)
    signal_mse = np.mean(
        (A_test @ (x_hat - x_true)) ** 2
    )
    estimation_error = np.linalg.norm(
        x_hat - x_true
    )

    return {
        "mse": float(mse),
        "signal_mse": float(signal_mse),
        "estimation_error": float(estimation_error),
    }


def weight_metrics(w, contaminated, atol=1e-10):
    w = np.asarray(w, dtype=float)
    contaminated = np.asarray(
        contaminated,
        dtype=bool,
    )

    # Zero weights are treated as detected contaminated samples
    detected = np.isclose(
        w,
        0.0,
        atol=atol,
    )

    tp = np.sum(detected & contaminated)
    fp = np.sum(detected & ~contaminated)
    fn = np.sum(~detected & contaminated)

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else np.nan
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else np.nan
    )

    f1 = (
        2.0 * precision * recall
        / (precision + recall)
        if np.isfinite(precision)
        and np.isfinite(recall)
        and precision + recall > 0
        else np.nan
    )

    retained = 100.0 * np.mean(~detected)

    return {
        "retained": float(retained),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def evaluate_regression(
    A_test,
    b_test,
    x_hat,
    x_true,
    w,
    contaminated,
):
    metrics = prediction_metrics(
        A_test,
        b_test,
        x_hat,
        x_true,
    )

    metrics.update(
        weight_metrics(
            w,
            contaminated,
        )
    )

    return metrics