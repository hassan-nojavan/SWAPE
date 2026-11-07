import numpy as np

from weighted_neural_network import (
    initialize_params,
    copy_params,
    initialize_adam,
    forward,
    backward,
    adam_step,
    sample_losses,
    update_w,
)

def backward_l2(
    params,
    cache,
    y,
    w,
    old_params,
    c_x,
    softplus_beta,
    l2_weight_decay,
):
    grads = backward(
        params,
        cache,
        y,
        w,
        old_params,
        c_x,
        softplus_beta,
    )

    for name in ["W1", "W2", "W3"]:
        grads[name] += (
            l2_weight_decay
            * params[name]
        )

    return grads


def fit_neural_network_l2(
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
    l2_weight_decay,
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

                _, cache = forward(
                    params,
                    xb,
                    softplus_beta,
                )

                grads = backward_l2(
                    params,
                    cache,
                    yb,
                    wb,
                    old_params,
                    c_x,
                    softplus_beta,
                    l2_weight_decay,
                )

                params, state = adam_step(
                    params,
                    grads,
                    state,
                    lr,
                )

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