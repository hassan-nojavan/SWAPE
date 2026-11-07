import numpy as np

from weighted_neural_network import (
    initialize_params,
    copy_params,
    softplus,
    sigmoid,
    initialize_adam,
    adam_step,
    sample_losses,
    update_w,
)

def apply_dropout(
    h,
    rng,
    dropout_p,
):
    keep_prob = 1.0 - dropout_p

    mask = (
        rng.random(h.shape) < keep_prob
    ).astype(float) / keep_prob

    return h * mask, mask


def forward_dropout(
    params,
    X,
    softplus_beta,
    dropout_p,
    rng,
):
    z1 = X @ params["W1"] + params["b1"]
    h1 = softplus(z1, softplus_beta)

    h1_used, mask1 = apply_dropout(
        h1,
        rng,
        dropout_p,
    )

    z2 = (
        h1_used @ params["W2"]
        + params["b2"]
    )

    h2 = softplus(z2, softplus_beta)

    h2_used, mask2 = apply_dropout(
        h2,
        rng,
        dropout_p,
    )

    logits = (
        h2_used @ params["W3"]
        + params["b3"]
    ).reshape(-1)

    cache = {
        "X": X,
        "z1": z1,
        "h1_used": h1_used,
        "mask1": mask1,
        "z2": z2,
        "h2_used": h2_used,
        "mask2": mask2,
        "logits": logits,
    }

    return logits, cache


def backward_dropout(
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
    h1_used = cache["h1_used"]
    mask1 = cache["mask1"]

    z2 = cache["z2"]
    h2_used = cache["h2_used"]
    mask2 = cache["mask2"]

    logits = cache["logits"]

    n = X.shape[0]

    probs = sigmoid(logits)

    dlogits = (
        w * (probs - y) / n
    ).reshape(-1, 1)

    grads = {}

    grads["W3"] = h2_used.T @ dlogits
    grads["b3"] = np.sum(dlogits, axis=0)

    dh2_used = dlogits @ params["W3"].T
    dh2 = dh2_used * mask2

    dz2 = dh2 * sigmoid(
        softplus_beta * z2
    )

    grads["W2"] = h1_used.T @ dz2
    grads["b2"] = np.sum(dz2, axis=0)

    dh1_used = dz2 @ params["W2"].T
    dh1 = dh1_used * mask1

    dz1 = dh1 * sigmoid(
        softplus_beta * z1
    )

    grads["W1"] = X.T @ dz1
    grads["b1"] = np.sum(dz1, axis=0)

    # Proximal term
    for name in grads:
        grads[name] += (
            c_x
            * (params[name] - old_params[name])
        )

    return grads


def fit_neural_network_dropout(
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
    dropout_p,
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

                _, cache = forward_dropout(
                    params,
                    xb,
                    softplus_beta,
                    dropout_p,
                    rng,
                )

                grads = backward_dropout(
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

        # No dropout during weight evaluation
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