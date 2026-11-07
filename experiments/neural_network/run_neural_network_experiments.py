import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit

from weighted_neural_network import (
    fit_neural_network,
    predict_probs,
)
from weighted_neural_network_l2 import (
    fit_neural_network_l2,
)
from weighted_neural_network_dropout import (
    fit_neural_network_dropout,
)
from weighted_neural_network_early_stopping import (
    fit_neural_network_early_stop,
)
from generate_neural_network_data import (
    generate_neural_network_data,
)
from neural_network_utils import (
    standardize_train_test,
    standardize_with_scaler,
    evaluate_neural_network,
)


# Example experiment settings. Adjust or tune as needed.
N_TRAIN = 500
N_TEST = 5000
N_REPS = 1

ALPHAS = [0.25, 0.30, 0.35, 0.40, 0.45]
MOONS_NOISE = 0.25

INPUT_DIM = 2
HIDDEN_1 = 256
HIDDEN_2 = 256
SOFTPLUS_BETA = 5.0

N_OUTER = 10
INNER_EPOCHS = 100
BATCH_SIZE = 64
LR = 1e-3

GAMMA = 0.1

C_W = 0.5
C_X = 1e-5

L2_WEIGHT_DECAY = 1e-4
DROPOUT_P = 0.20

EARLY_STOP_VAL_FRAC = 0.20
EARLY_STOP_PATIENCE = 30
EARLY_STOP_MIN_DELTA = 1e-4

BASE_SEED = 123


def fit_standard(
    X_train,
    y_train,
    seed,
):
    return fit_neural_network(
        X=X_train,
        y=y_train,
        gamma=GAMMA,
        c_w=C_W,
        c_x=C_X,
        input_dim=INPUT_DIM,
        hidden_1=HIDDEN_1,
        hidden_2=HIDDEN_2,
        softplus_beta=SOFTPLUS_BETA,
        n_outer=N_OUTER,
        inner_epochs=INNER_EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        seed=seed,
    )


def fit_l2(
    X_train,
    y_train,
    seed,
):
    return fit_neural_network_l2(
        X=X_train,
        y=y_train,
        gamma=GAMMA,
        c_w=C_W,
        c_x=C_X,
        input_dim=INPUT_DIM,
        hidden_1=HIDDEN_1,
        hidden_2=HIDDEN_2,
        softplus_beta=SOFTPLUS_BETA,
        n_outer=N_OUTER,
        inner_epochs=INNER_EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        l2_weight_decay=L2_WEIGHT_DECAY,
        seed=seed,
    )


def fit_dropout(
    X_train,
    y_train,
    seed,
):
    return fit_neural_network_dropout(
        X=X_train,
        y=y_train,
        gamma=GAMMA,
        c_w=C_W,
        c_x=C_X,
        input_dim=INPUT_DIM,
        hidden_1=HIDDEN_1,
        hidden_2=HIDDEN_2,
        softplus_beta=SOFTPLUS_BETA,
        n_outer=N_OUTER,
        inner_epochs=INNER_EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        dropout_p=DROPOUT_P,
        seed=seed,
    )


def fit_early_stopping(
    X_train,
    y_train,
    contaminated,
    seed,
):
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=EARLY_STOP_VAL_FRAC,
        random_state=seed,
    )

    fit_idx, val_idx = next(
        splitter.split(
            X_train,
            y_train,
        )
    )

    X_fit_raw = X_train[fit_idx]
    X_val_raw = X_train[val_idx]

    y_fit = y_train[fit_idx]
    y_val = y_train[val_idx]

    X_fit, X_val, scaler = standardize_train_test(
        X_fit_raw,
        X_val_raw,
    )

    params, w, meta = fit_neural_network_early_stop(
        X=X_fit,
        y=y_fit,
        X_val=X_val,
        y_val=y_val,
        gamma=GAMMA,
        c_w=C_W,
        c_x=C_X,
        input_dim=INPUT_DIM,
        hidden_1=HIDDEN_1,
        hidden_2=HIDDEN_2,
        softplus_beta=SOFTPLUS_BETA,
        n_outer=N_OUTER,
        inner_epochs=INNER_EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        patience=EARLY_STOP_PATIENCE,
        min_delta=EARLY_STOP_MIN_DELTA,
        seed=seed,
    )

    return (
        params,
        w,
        scaler,
        contaminated[fit_idx],
        meta,
    )


def run_replication(rep, alpha):
    data_seed = (
        BASE_SEED
        + 10000 * rep
        + int(alpha * 1000)
    )

    (
        X_train_raw,
        y_train,
        _,
        X_test_raw,
        y_test,
        contaminated,
    ) = generate_neural_network_data(
        n_train=N_TRAIN,
        n_test=N_TEST,
        noise=MOONS_NOISE,
        alpha=alpha,
        seed=data_seed,
    )

    results = []

    for variant in [
        "none",
        "l2",
        "dropout",
    ]:
        X_train, X_test, _ = standardize_train_test(
            X_train_raw,
            X_test_raw,
        )

        seed = data_seed + 999

        if variant == "none":
            params, w = fit_standard(
                X_train,
                y_train,
                seed,
            )

        elif variant == "l2":
            params, w = fit_l2(
                X_train,
                y_train,
                seed,
            )

        else:
            params, w = fit_dropout(
                X_train,
                y_train,
                seed,
            )

        probs = predict_probs(
            params,
            X_test,
            SOFTPLUS_BETA,
        )

        metrics = evaluate_neural_network(
            y_test=y_test,
            probs=probs,
            w=w,
            contaminated=contaminated,
        )

        results.append({
            "rep": rep,
            "alpha": alpha,
            "variant": variant,
            **metrics,
        })

    seed = data_seed + 999

    (
        params,
        w,
        scaler,
        contaminated_fit,
        meta,
    ) = fit_early_stopping(
        X_train_raw,
        y_train,
        contaminated,
        seed,
    )

    X_test = standardize_with_scaler(
        X_test_raw,
        scaler,
    )

    probs = predict_probs(
        params,
        X_test,
        SOFTPLUS_BETA,
    )

    metrics = evaluate_neural_network(
        y_test=y_test,
        probs=probs,
        w=w,
        contaminated=contaminated_fit,
    )

    results.append({
        "rep": rep,
        "alpha": alpha,
        "variant": "early_stopping",
        "epochs_used": meta["epochs_used"],
        "stopped_early": meta["stopped_early"],
        **metrics,
    })

    return results


def summarize_results(results):
    return (
        results
        .groupby(
            [
                "variant",
                "alpha",
            ],
            as_index=False,
        )
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            retained_mean=("retained", "mean"),
            retained_std=("retained", "std"),
            precision_mean=("precision", "mean"),
            precision_std=("precision", "std"),
            recall_mean=("recall", "mean"),
            recall_std=("recall", "std"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
        )
    )


rows = []

for rep in range(N_REPS):
    for alpha in ALPHAS:
        rows.extend(
            run_replication(
                rep,
                alpha,
            )
        )

results = pd.DataFrame(rows)
summary = summarize_results(results)

results.to_csv(
    "neural_network_results.csv",
    index=False,
)

summary.to_csv(
    "neural_network_summary.csv",
    index=False,
)

print(summary)