import numpy as np


def train_test_split(A, test_fraction, seed):
    rng = np.random.default_rng(seed)

    A_train = A.copy()
    A_test = np.zeros_like(A)

    for i in range(A.shape[0]):
        rated = np.flatnonzero(A[i] > 0)

        if len(rated) < 2:
            continue

        n_test = max(
            1,
            int(round(test_fraction * len(rated))),
        )
        n_test = min(n_test, len(rated) - 1)

        test_idx = rng.choice(
            rated,
            size=n_test,
            replace=False,
        )

        A_test[i, test_idx] = A[i, test_idx]
        A_train[i, test_idx] = 0.0

    return A_train, A_test


def mae(A_test, A_hat):
    observed = A_test > 0

    if not np.any(observed):
        return np.nan

    return float(
        np.mean(
            np.abs(
                A_test[observed]
                - A_hat[observed]
            )
        )
    )


def ranking_metrics(
    A_train,
    A_test,
    A_hat,
    k,
    relevant_threshold,
):
    n_users = A_train.shape[0]

    hits = []
    reciprocal_ranks = []

    for i in range(n_users):
        scores = A_hat[i].copy()

        # Recommend only items outside the training set
        scores[A_train[i] > 0] = -np.inf

        n_candidates = np.sum(
            np.isfinite(scores)
        )
        k_user = min(k, n_candidates)

        if k_user == 0:
            hits.append(0.0)
            reciprocal_ranks.append(0.0)
            continue

        recommended = np.argsort(
            scores
        )[-k_user:][::-1]

        relevant = np.flatnonzero(
            A_test[i] >= relevant_threshold
        )

        ranks = np.flatnonzero(
            np.isin(recommended, relevant)
        )

        if len(ranks) > 0:
            hits.append(1.0)
            reciprocal_ranks.append(
                1.0 / (ranks[0] + 1)
            )
        else:
            hits.append(0.0)
            reciprocal_ranks.append(0.0)

    return {
        "hit_rate": float(np.mean(hits)),
        "mrr": float(np.mean(reciprocal_ranks)),
    }


def retained_percentage(W, atol=1e-8):
    return float(
        100.0 * np.mean(W > atol)
    )


def evaluate_recommender(
    A_train,
    A_test,
    A_hat,
    W,
    k,
    relevant_threshold,
):
    metrics = {
        "mae": mae(
            A_test,
            A_hat,
        ),
        "retained": retained_percentage(W),
    }

    metrics.update(
        ranking_metrics(
            A_train,
            A_test,
            A_hat,
            k,
            relevant_threshold,
        )
    )

    return metrics