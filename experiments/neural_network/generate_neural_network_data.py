import numpy as np
from sklearn.datasets import make_moons


def generate_neural_network_data(
    n_train,
    n_test,
    noise,
    alpha,
    seed,
):
    X_train, y_train_clean = make_moons(
        n_samples=n_train,
        noise=noise,
        random_state=seed,
    )

    X_test, y_test = make_moons(
        n_samples=n_test,
        noise=noise,
        random_state=seed + 999,
    )

    rng = np.random.default_rng(seed + 777)

    n_contaminated = int(alpha * n_train)
    contaminated_idx = rng.choice(
        n_train,
        size=n_contaminated,
        replace=False,
    )

    y_train = y_train_clean.copy()
    y_train[contaminated_idx] = (
        1 - y_train[contaminated_idx]
    )

    contaminated = np.zeros(
        n_train,
        dtype=bool,
    )
    contaminated[contaminated_idx] = True

    return (
        X_train.astype(float),
        y_train.astype(float),
        y_train_clean.astype(float),
        X_test.astype(float),
        y_test.astype(float),
        contaminated,
    )
