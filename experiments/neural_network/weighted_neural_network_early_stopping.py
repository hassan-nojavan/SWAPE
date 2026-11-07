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

def validation_score(
    params,
    X_val,
    y_val,
    softplus_beta,
):
    loss = sample_losses(
        params,
        X_val,
        y_val,
        softplus_beta,
    )

    return -float(np.mean(loss))


def fit_neural_network_early_stop(
    X,
    y,
    X_val,
    y_val,
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
    patience,
    min_delta,
    seed,
):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()

    X_val = np.asarray(
        X_val,
        dtype=float,
    )

    y_val = np.asarray(
        y_val,
        dtype=float,
    ).ravel()

    rng = np.random.default_rng(seed)

    params = initialize_params(
        input_dim,
        hidden_1,
        hidden_2,
        seed,
    )

    n = X.shape[0]
    w = np.ones(n)

    best_params = copy_params(params)
    best_w = w.copy()

    best_score = -np.inf
    best_epoch = 0

    patience_count = 0
    global_epoch = 0

    for _ in range(n_outer):

        old_params = copy_params(params)
        state = initialize_adam(params)

        for _ in range(inner_epochs):

            global_epoch += 1

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

            score = validation_score(
                params,
                X_val,
                y_val,
                softplus_beta,
            )

            if score > best_score + min_delta:

                best_score = score
                best_params = copy_params(params)
                best_w = w.copy()

                best_epoch = global_epoch
                patience_count = 0

            else:
                patience_count += 1

            if patience_count >= patience:

                meta = {
                    "epochs_used": best_epoch,
                    "stopped_early": True,
                    "best_es_score": best_score,
                }

                return (
                    best_params,
                    best_w,
                    meta,
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

        score = validation_score(
            params,
            X_val,
            y_val,
            softplus_beta,
        )

        if score > best_score + min_delta:

            best_score = score
            best_params = copy_params(params)
            best_w = w.copy()

            best_epoch = global_epoch
            patience_count = 0

    meta = {
        "epochs_used": best_epoch,
        "stopped_early": False,
        "best_es_score": best_score,
    }

    return (
        best_params,
        best_w,
        meta,
    )