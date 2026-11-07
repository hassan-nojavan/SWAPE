import numpy as np


def initialize_params(input_dim, hidden_1, hidden_2, seed):
    rng = np.random.default_rng(seed)

    params = {
        "W1": rng.normal(
            0.0,
            np.sqrt(2.0 / input_dim),
            size=(input_dim, hidden_1),
        ),
        "b1": np.zeros(hidden_1),

        "W2": rng.normal(
            0.0,
            np.sqrt(2.0 / hidden_1),
            size=(hidden_1, hidden_2),
        ),
        "b2": np.zeros(hidden_2),

        "W3": rng.normal(
            0.0,
            np.sqrt(2.0 / hidden_2),
            size=(hidden_2, 1),
        ),
        "b3": np.zeros(1),
    }

    return params


def copy_params(params):
    return {
        name: value.copy()
        for name, value in params.items()
    }


def softplus(z, beta):
    return np.logaddexp(0.0, beta * z) / beta


def sigmoid(z):
    z = np.asarray(z)
    out = np.empty_like(z, dtype=float)

    positive = z >= 0
    negative = ~positive

    out[positive] = 1.0 / (
        1.0 + np.exp(-z[positive])
    )

    exp_z = np.exp(z[negative])

    out[negative] = exp_z / (
        1.0 + exp_z
    )

    return out


def forward(params, X, softplus_beta):
    z1 = X @ params["W1"] + params["b1"]
    h1 = softplus(z1, softplus_beta)

    z2 = h1 @ params["W2"] + params["b2"]
    h2 = softplus(z2, softplus_beta)

    logits = (
        h2 @ params["W3"] + params["b3"]
    ).reshape(-1)

    cache = {
        "X": X,
        "z1": z1,
        "h1": h1,
        "z2": z2,
        "h2": h2,
        "logits": logits,
    }

    return logits, cache


def bce_loss(logits, y):
    return (
        np.maximum(logits, 0.0)
        - logits * y
        + np.log1p(np.exp(-np.abs(logits)))
    )


def sample_losses(
    params,
    X,
    y,
    softplus_beta,
):
    logits, _ = forward(
        params,
        X,
        softplus_beta,
    )

    return bce_loss(logits, y)


def backward(
    params,
    cache,
    y,
    w,
    old_params,
    c_x,
    softplus_beta,
):
    X = cache["X"]
    z1 = cache["z1"]
    h1 = cache["h1"]

    z2 = cache["z2"]
    h2 = cache["h2"]

    logits = cache["logits"]

    n = X.shape[0]

    probs = sigmoid(logits)

    dlogits = (
        w * (probs - y) / n
    ).reshape(-1, 1)

    grads = {}

    grads["W3"] = h2.T @ dlogits
    grads["b3"] = np.sum(dlogits, axis=0)

    dh2 = dlogits @ params["W3"].T
    dz2 = dh2 * sigmoid(softplus_beta * z2)

    grads["W2"] = h1.T @ dz2
    grads["b2"] = np.sum(dz2, axis=0)

    dh1 = dz2 @ params["W2"].T
    dz1 = dh1 * sigmoid(softplus_beta * z1)

    grads["W1"] = X.T @ dz1
    grads["b1"] = np.sum(dz1, axis=0)

    # Proximal term
    for name in grads:
        grads[name] += (
            c_x
            * (params[name] - old_params[name])
        )

    return grads


def initialize_adam(params):
    state = {}

    for name, value in params.items():
        state[name] = {
            "m": np.zeros_like(value),
            "v": np.zeros_like(value),
        }

    state["t"] = 0

    return state


def adam_step(
    params,
    grads,
    state,
    lr,
    beta1=0.9,
    beta2=0.999,
    eps=1e-8,
):
    state["t"] += 1
    t = state["t"]

    for name in params:
        state[name]["m"] = (
            beta1 * state[name]["m"]
            + (1.0 - beta1) * grads[name]
        )

        state[name]["v"] = (
            beta2 * state[name]["v"]
            + (1.0 - beta2) * grads[name] ** 2
        )

        m_hat = (
            state[name]["m"]
            / (1.0 - beta1**t)
        )

        v_hat = (
            state[name]["v"]
            / (1.0 - beta2**t)
        )

        params[name] -= (
            lr
            * m_hat
            / (np.sqrt(v_hat) + eps)
        )

    return params, state


def update_w(loss, w_prev, gamma, c_w):
    z = w_prev - loss / c_w
    tau = gamma / c_w

    w = np.empty_like(w_prev)

    left = z < 1.0 - tau

    middle = (
        (z >= 1.0 - tau)
        & (z <= 1.0 + tau)
    )

    right = z > 1.0 + tau

    # Closed-form weight update
    w[left] = np.maximum(
        0.0,
        z[left] + tau,
    )

    w[middle] = 1.0
    w[right] = z[right] - tau

    return np.maximum(w, 0.0)


def predict_probs(
    params,
    X,
    softplus_beta,
):
    logits, _ = forward(
        params,
        X,
        softplus_beta,
    )

    return sigmoid(logits)


def fit_neural_network(
    X,
    y,
    gamma,
    c_w,
    c_x,
    input_dim,
    hidden_1,
    hidden_2,
    softplus_beta,
    n_outer,
    inner_epochs,
    batch_size,
    lr,
    seed,
):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()

    rng = np.random.default_rng(seed)

    params = initialize_params(
        input_dim,
        hidden_1,
        hidden_2,
        seed,
    )

    n = X.shape[0]
    w = np.ones(n)

    for _ in range(n_outer):

        old_params = copy_params(params)
        state = initialize_adam(params)

        # Update network parameters
        for _ in range(inner_epochs):

            indices = rng.permutation(n)

            for start in range(
                0,
                n,
                batch_size,
            ):
                idx = indices[
                    start:start + batch_size
                ]

                xb = X[idx]
                yb = y[idx]
                wb = w[idx]

                _, cache = forward(
                    params,
                    xb,
                    softplus_beta,
                )

                grads = backward(
                    params,
                    cache,
                    yb,
                    wb,
                    old_params,
                    c_x,
                    softplus_beta,
                )

                params, state = adam_step(
                    params,
                    grads,
                    state,
                    lr,
                )

        # Update sample weights
        loss = sample_losses(
            params,
            X,
            y,
            softplus_beta,
        )

        w = update_w(
            loss,
            w,
            gamma,
            c_w,
        )

    return params, w