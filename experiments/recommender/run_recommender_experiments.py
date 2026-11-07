import pandas as pd

from weighted_recommender import fit_recommender
from generate_recommender_data import generate_recommender_data
from recommender_utils import (
    train_test_split,
    evaluate_recommender,
)


# Example experiment settings. Adjust or tune as needed.
N_USERS = 100
N_ITEMS = 50
DENSITY = 0.25
N_REPS = 1

TEST_FRACTION = 0.20
K_VALUES = [5, 10, 15]
RELEVANT_THRESHOLD = 4.0

LAMBDA_X = 0.05
GAMMA = 0.1
THETA = 0.5
ETA = 2.0
C_X = 600000
C_W = 0.1

MAX_ITER = 100
TOL = 1e-4

DATA_SEED = 2026


def run_replication(rep):
    seed = DATA_SEED + rep

    A = generate_recommender_data(
        n_users=N_USERS,
        n_items=N_ITEMS,
        density=DENSITY,
        seed=seed,
    )

    A_train, A_test = train_test_split(
        A,
        test_fraction=TEST_FRACTION,
        seed=seed,
    )

    X, W, A_hat, n_iters = fit_recommender(
        A=A_train,
        lambda_x=LAMBDA_X,
        gamma=GAMMA,
        theta=THETA,
        eta=ETA,
        c_x=C_X,
        c_w=C_W,
        max_iter=MAX_ITER,
        tol=TOL,
    )

    results = []

    for k in K_VALUES:
        metrics = evaluate_recommender(
            A_train=A_train,
            A_test=A_test,
            A_hat=A_hat,
            W=W,
            k=k,
            relevant_threshold=RELEVANT_THRESHOLD,
        )

        results.append({
            "rep": rep,
            "k": k,
            "avg_iterations": n_iters.mean(),
            **metrics,
        })

    return results


def summarize_results(results):
    return (
        results
        .groupby(
            "k",
            as_index=False,
        )
        .agg(
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            hit_rate_mean=("hit_rate", "mean"),
            hit_rate_std=("hit_rate", "std"),
            mrr_mean=("mrr", "mean"),
            mrr_std=("mrr", "std"),
            retained_mean=("retained", "mean"),
            retained_std=("retained", "std"),
            avg_iterations=("avg_iterations", "mean"),
        )
    )


rows = []

for rep in range(N_REPS):
    rows.extend(
        run_replication(rep)
    )

results = pd.DataFrame(rows)
summary = summarize_results(results)

results.to_csv(
    "recommender_results.csv",
    index=False,
)

summary.to_csv(
    "recommender_summary.csv",
    index=False,
)

print(summary)