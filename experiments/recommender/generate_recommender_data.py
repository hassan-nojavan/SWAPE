import numpy as np


# The paper uses the Netflix Prize dataset.
# This generator creates a small synthetic rating matrix for code examples.
def generate_recommender_data(
    n_users=24,
    n_items=12,
    density=0.6,
    seed=2026,
):
    rng = np.random.default_rng(seed)

    ratings = rng.integers(
        1,
        6,
        size=(n_users, n_items),
    )

    observed = (
        rng.random((n_users, n_items))
        < density
    )

    # Keep at least one observed rating for each user and item
    for i in range(n_users):
        if not observed[i].any():
            observed[
                i,
                rng.integers(n_items),
            ] = True

    for j in range(n_items):
        if not observed[:, j].any():
            observed[
                rng.integers(n_users),
                j,
            ] = True

    A = ratings * observed

    return A.astype(float)