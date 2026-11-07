import numpy as np
import pandas as pd

from weighted_regression import fit_regression
from generate_regression_data import (
    generate_clean_data,
    generate_vertical_data,
    generate_interior_clustered_data,
)
from regression_utils import evaluate_regression


# Example experiment settings. Adjust or tune as needed.
N_TRAIN = 600
N_TEST = 5000
N_REPS = 1

ALPHAS = [0.2, 0.3, 0.4]
TAUS = [5.0, 8.0]
N_CLOUDS = [1, 2]

GAMMA = 0.1
THETA = 1.0
ETA = 3.0
C_W = 0.2
C_X = 1e-6

MAX_ITER = 1000
TOL = 1e-6

DATA_SEED = 2025
INIT_SEED = 2025


def fit_and_evaluate(
    A_train,
    b_train,
    A_test,
    b_test,
    x_true,
    contaminated,
    seed,
):
    x_hat, w_hat, n_iter, converged = fit_regression(
        A=A_train,
        b=b_train,
        gamma=GAMMA,
        theta=THETA,
        eta=ETA,
        c_w=C_W,
        c_x=C_X,
        max_iter=MAX_ITER,
        tol=TOL,
        seed=seed,
    )

    metrics = evaluate_regression(
        A_test=A_test,
        b_test=b_test,
        x_hat=x_hat,
        x_true=x_true,
        w=w_hat,
        contaminated=contaminated,
    )

    metrics["n_iter"] = n_iter
    metrics["converged"] = converged

    return metrics


def run_replication(rep):
    rng = np.random.default_rng(DATA_SEED + rep)
    seed = INIT_SEED + rep

    results = []

    A_train, b_train, x_true, contaminated = (
        generate_clean_data(
            N_TRAIN,
            rng,
        )
    )

    A_test, b_test, _, _ = generate_clean_data(
        N_TEST,
        rng,
    )

    metrics = fit_and_evaluate(
        A_train,
        b_train,
        A_test,
        b_test,
        x_true,
        contaminated,
        seed,
    )

    results.append({
        "rep": rep,
        "setting": "clean",
        "alpha": 0.0,
        "tau": 0.0,
        "n_clouds": 0,
        **metrics,
    })

    for alpha in ALPHAS:
        for tau in TAUS:

            A_train, b_train, x_true, contaminated = (
                generate_vertical_data(
                    N_TRAIN,
                    alpha,
                    tau,
                    rng,
                )
            )

            metrics = fit_and_evaluate(
                A_train,
                b_train,
                A_test,
                b_test,
                x_true,
                contaminated,
                seed,
            )

            results.append({
                "rep": rep,
                "setting": "vertical",
                "alpha": alpha,
                "tau": tau,
                "n_clouds": 0,
                **metrics,
            })

            for n_clouds in N_CLOUDS:

                A_train, b_train, x_true, contaminated = (
                    generate_interior_clustered_data(
                        N_TRAIN,
                        alpha,
                        tau,
                        n_clouds,
                        rng,
                    )
                )

                metrics = fit_and_evaluate(
                    A_train,
                    b_train,
                    A_test,
                    b_test,
                    x_true,
                    contaminated,
                    seed,
                )

                results.append({
                    "rep": rep,
                    "setting": "interior_clustered",
                    "alpha": alpha,
                    "tau": tau,
                    "n_clouds": n_clouds,
                    **metrics,
                })

    return results


def summarize_results(results):
    return (
        results
        .groupby(
            [
                "setting",
                "alpha",
                "tau",
                "n_clouds",
            ],
            as_index=False,
        )
        .agg(
            mse_mean=("mse", "mean"),
            mse_std=("mse", "std"),
            signal_mse_mean=("signal_mse", "mean"),
            signal_mse_std=("signal_mse", "std"),
            estimation_error_mean=(
                "estimation_error",
                "mean",
            ),
            estimation_error_std=(
                "estimation_error",
                "std",
            ),
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


def main():
    rows = []

    for rep in range(1, N_REPS + 1):
        rows.extend(
            run_replication(rep)
        )

    results = pd.DataFrame(rows)
    summary = summarize_results(results)

    results.to_csv(
        "regression_results.csv",
        index=False,
    )

    summary.to_csv(
        "regression_summary.csv",
        index=False,
    )

    print(summary)


main()