import numpy as np


def mcp_penalty(w, theta, eta):
    z = np.abs(w - 1.0)
    in_band = z <= eta * theta

    return np.where(
        in_band,
        theta * z - z**2 / (2.0 * eta),
        eta * theta**2 / 2.0,
    )


def objective(A, a, x, w, lambda_x, gamma, theta, eta):
    residual = a - A @ x

    return float(
        0.5 * np.sum(w * residual**2)
        + lambda_x * np.sum(x)
        + gamma * np.sum(mcp_penalty(w, theta, eta))
    )


def update_x(A, a, w, x_prev, j, lambda_x, c_x):
    gradient = A.T @ (w * (A @ x_prev - a))

    z = x_prev - gradient / c_x
    threshold = lambda_x / c_x

    x = np.maximum(z - threshold, 0.0)
    x[j] = 0.0

    return x


def update_w(
    residual,
    w_prev,
    gamma,
    theta,
    eta,
    c_w,
):
    t = w_prev - residual**2 / (2.0 * c_w)

    denominator = c_w - gamma / eta

    lower = 1.0 - eta * theta
    upper = 1.0 + eta * theta

    t_left = 1.0 - gamma * theta / c_w
    t_right = 1.0 + gamma * theta / c_w

    if eta * theta < 1.0:
        t_zero = 0.0
    else:
        t_zero = (
            gamma / c_w
        ) * (1.0 / eta - theta)

    w = np.empty_like(t)

    m0 = t <= t_zero
    w[m0] = 0.0

    if eta * theta < 1.0:
        m1 = (t > 0.0) & (t <= lower)
        w[m1] = t[m1]

        m2 = (t > lower) & (t < t_left)
    else:
        m2 = (t > t_zero) & (t < t_left)

    w[m2] = 1.0 - (
        c_w * (1.0 - t[m2]) - gamma * theta
    ) / denominator

    m3 = (t >= t_left) & (t <= t_right)
    w[m3] = 1.0

    m4 = (t > t_right) & (t < upper)
    w[m4] = 1.0 + (
        c_w * (t[m4] - 1.0) - gamma * theta
    ) / denominator

    m5 = t >= upper
    w[m5] = t[m5]

    return np.maximum(w, 0.0)


def fit_column(
    A,
    j,
    lambda_x,
    gamma,
    theta,
    eta,
    c_x,
    c_w,
    max_iter,
    tol,
):
    m, n = A.shape
    a = A[:, j].astype(float)

    if np.count_nonzero(a) == 0:
        x = np.zeros(n)
        w = np.ones(m)
        return x, w, 0

    x = np.zeros(n)
    w = np.ones(m)

    obj_prev = np.inf
    n_iter = 0

    for iteration in range(1, max_iter + 1):

        w_used = w.copy()

        x = update_x(
            A,
            a,
            w,
            x,
            j,
            lambda_x,
            c_x,
        )

        residual = a - A @ x

        w = update_w(
            residual,
            w,
            gamma,
            theta,
            eta,
            c_w,
        )

        obj = objective(
            A,
            a,
            x,
            w,
            lambda_x,
            gamma,
            theta,
            eta,
        )

        n_iter = iteration

        if np.isfinite(obj_prev):
            rel_change = abs(obj_prev - obj) / max(
                abs(obj),
                np.finfo(float).tiny,
            )

            if rel_change <= tol:
                break

        obj_prev = obj

    return x, w_used, n_iter


def fit_recommender(
    A,
    lambda_x,
    gamma,
    theta,
    eta,
    c_x,
    c_w,
    max_iter,
    tol,
):
    A = np.asarray(A, dtype=float)

    if c_w <= gamma / eta:
        raise ValueError(
            "c_w must be greater than gamma / eta."
        )

    _, n = A.shape

    results = []

    for j in range(n):
        result = fit_column(
            A,
            j,
            lambda_x,
            gamma,
            theta,
            eta,
            c_x,
            c_w,
            max_iter,
            tol,
        )

        results.append(result)

    X = np.column_stack(
        [result[0] for result in results]
    )

    W = np.column_stack(
        [result[1] for result in results]
    )

    n_iters = np.array(
        [result[2] for result in results]
    )

    A_hat = A @ X

    return X, W, A_hat, n_iters