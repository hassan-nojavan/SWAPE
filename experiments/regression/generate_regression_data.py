import numpy as np


D = 6
MU_A = 7.5
SIGMA_A = 4.0
SIGMA_E = 1.0
X_TRUE_VALUE = 5.0


def generate_design(n, rng):
    return rng.normal(
        MU_A,
        SIGMA_A,
        size=(n, D),
    )


def generate_clean_data(n, rng):
    A = generate_design(n, rng)
    x_true = np.full(D, X_TRUE_VALUE)
    epsilon = rng.normal(0.0, SIGMA_E, size=n)

    b = A @ x_true + epsilon
    contaminated = np.zeros(n, dtype=bool)

    return A, b, x_true, contaminated


def generate_vertical_data(n, alpha, tau, rng):
    A = generate_design(n, rng)
    x_true = np.full(D, X_TRUE_VALUE)

    epsilon = rng.normal(0.0, SIGMA_E, size=n)
    contaminated = np.zeros(n, dtype=bool)

    n_contaminated = int(round(alpha * n))
    indices = rng.choice(
        n,
        size=n_contaminated,
        replace=False,
    )

    epsilon[indices] = rng.normal(
        tau,
        SIGMA_E,
        size=n_contaminated,
    )
    contaminated[indices] = True

    b = A @ x_true + epsilon

    return A, b, x_true, contaminated


def generate_interior_clustered_data(
    n,
    alpha,
    tau,
    n_clouds,
    rng,
):
    A = generate_design(n, rng)
    x_true = np.full(D, X_TRUE_VALUE)

    epsilon = rng.normal(0.0, SIGMA_E, size=n)
    contaminated = np.zeros(n, dtype=bool)

    n_contaminated = int(round(alpha * n))
    indices = rng.choice(
        n,
        size=n_contaminated,
        replace=False,
    )

    center_1 = np.median(A[:3], axis=0)
    center_2 = np.median(A[-3:], axis=0)

    if n_clouds == 1:
        A[indices] = rng.uniform(
            center_1,
            center_1 + 0.25,
            size=(n_contaminated, D),
        )

        epsilon[indices] = rng.normal(
            tau,
            SIGMA_E,
            size=n_contaminated,
        )

    elif n_clouds == 2:
        split = n_contaminated // 2
        indices_1 = indices[:split]
        indices_2 = indices[split:]

        A[indices_1] = rng.uniform(
            center_1,
            center_1 + 0.25,
            size=(len(indices_1), D),
        )

        A[indices_2] = rng.uniform(
            center_2,
            center_2 + 0.25,
            size=(len(indices_2), D),
        )

        epsilon[indices_1] = rng.normal(
            tau,
            SIGMA_E,
            size=len(indices_1),
        )

        epsilon[indices_2] = rng.normal(
            -tau,
            SIGMA_E,
            size=len(indices_2),
        )

    else:
        raise ValueError("n_clouds must be 1 or 2.")

    contaminated[indices] = True

    b = A @ x_true + epsilon

    return A, b, x_true, contaminated