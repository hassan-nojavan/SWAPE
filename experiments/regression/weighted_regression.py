import numpy as np


def initialize(n, p, seed):
    rng = np.random.default_rng(seed)

    x = rng.normal(0.0, 0.1, size=p)
    w = rng.uniform(0.2, 1.0, size=n)

    return x, w


def update_x(A, b, w, x_prev, c_x):
    WA = A * w[:, None]

    lhs = A.T @ WA + (c_x / 2.0) * np.eye(A.shape[1])
    rhs = A.T @ (w * b) + (c_x / 2.0) * x_prev

    return np.linalg.solve(lhs, rhs)


def mcp_penalty(w, theta, eta):
    z = np.abs(w - 1.0)
    in_band = z <= eta * theta

    return (
        np.sum(theta * z[in_band] - z[in_band] ** 2 / (2.0 * eta))
        + np.sum((eta * theta**2 / 2.0) * (~in_band))
    )


def update_w(w_prev, residual_sq, gamma, theta, eta, c_w):
    lower = max(0.0, 1.0 - eta * theta)
    upper = 1.0 + eta * theta

    t = w_prev - residual_sq / c_w
    w = np.empty_like(t)

    t_left = 1.0 - gamma * theta / c_w
    t_right = 1.0 + gamma * theta / c_w
    t_zero = (gamma / c_w) * (1.0 / eta - theta)

    if lower > 0.0:
        m0 = t <= 0.0
        m1 = (t > 0.0) & (t <= lower)
        m2 = (t > lower) & (t < t_left)
    else:
        m0 = t <= t_zero
        m1 = np.zeros_like(t, dtype=bool)
        m2 = (t > t_zero) & (t < t_left)

    m3 = (t >= t_left) & (t <= t_right)
    m4 = (t > t_right) & (t < upper)
    m5 = t >= upper

    denominator = c_w - gamma / eta

    if abs(denominator) < 1e-12:
        denominator = 1e-12 if denominator >= 0.0 else -1e-12

    # Closed-form weight update
    w[m0] = 0.0

    if lower > 0.0:
        w[m1] = t[m1]

    w[m2] = 1.0 - (
        c_w * (1.0 - t[m2]) - gamma * theta
    ) / denominator

    w[m3] = 1.0

    w[m4] = 1.0 + (
        c_w * (t[m4] - 1.0) - gamma * theta
    ) / denominator

    w[m5] = t[m5]

    return np.maximum(w, 0.0)


def objective(A, b, x, w, gamma, theta, eta):
    residual = b - A @ x

    return float(
        np.dot(w, residual**2)
        + gamma * mcp_penalty(w, theta, eta)
    )


def fit_regression(
    A,
    b,
    gamma,
    theta,
    eta,
    c_w,
    c_x,
    max_iter,
    tol,
    seed,
):
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).ravel()

    n, p = A.shape
    x, w = initialize(n, p, seed)

    obj_prev = objective(
        A, b, x, w, gamma, theta, eta
    )

    converged = False
    n_iter = 0

    for iteration in range(1, max_iter + 1):

        # Update model parameters
        x = update_x(
            A, b, w, x, c_x
        )

        # Update sample weights
        residual = b - A @ x

        w = update_w(
            w,
            residual**2,
            gamma,
            theta,
            eta,
            c_w,
        )

        obj = objective(
            A, b, x, w, gamma, theta, eta
        )

        rel_change = abs(obj - obj_prev) / (
            abs(obj) + 1e-16
        )

        n_iter = iteration

        if rel_change < tol:
            converged = True
            break

        obj_prev = obj

    return x, w, n_iter, converged