import numpy as np
from numpy import random
import tensorly as tl
from tensorly.decomposition import non_negative_parafac


def check_inputs(X):
    assert X.ndim > 1
    assert min(X.shape) > 1
    return X


def predict_seasonal_tensor(factors, n_sample, t=0):
    n_fold = n_sample // len(factors[0]) + 1
    pred = tl.kruskal_to_tensor((None, factors))
    pred = np.tile(pred, (n_fold, *[1] * (len(factors) - 1)))
    return pred[:n_sample]


def init_seasonal_factors(tensor, rank, period, t=0,
    n_iter_max=100, tol=10e-7, random_state=None):

    if period == 0:
        return None

    else:
        cyclic_mean_tensor = np.zeros((period, *tensor.shape[1:]))
        n_sample    = tensor.shape[0]
        n_section   = n_sample // period
        season_ids  = np.arange(t, t + n_sample, 1) % period
        diff_ids    = n_sample - period * n_section
        start_point = np.where(season_ids==0)[0][0]

        rolled_tensor = np.roll(tensor, -start_point, axis=0)
        if diff_ids > 1:
            rolled_tensor = rolled_tensor[:-diff_ids]

        for w in range(0, n_sample - period, period):
            one_period = rolled_tensor[w * period: (w+1) * period]
            cyclic_mean_tensor += one_period - one_period.mean(axis=0)

        _, factors = non_negative_parafac(
            cyclic_mean_tensor, rank,
            n_iter_max=n_iter_max, tol=tol,
            random_state=random_state)

        return factors


def init_interaction_factors(tensor, rank, n_iter_max=100, random_state=None):
    _, factors = non_negative_parafac(
        tensor, rank, n_iter_max=n_iter_max, random_state=random_state)
    return factors


def initialize(X, n_dim_c, n_dim_s, n_season, t):
    # Fully-linear initialization
    S = init_seasonal_factors(X, n_dim_s, n_season, t)
    Y = predict_seasonal_tensor(S, len(X), t=t)
    C = init_interaction_factors(X - Y, n_dim_c, n_season, t)
    Z = tl.kruskal_to_tensor((None, C))
    return S, C, Y, Z


def solve_cosmo(X, n_dim_c, n_dim_s=0, n_season=0, t=0,
    max_iter=10):

    """
        X: A tensor timeseries
        t: The first timestamp of X
    """

    X = check_inputs(X)
    n_sample = X.shape[0]
    n_modes = X.ndim
    n_dims = X.shape[1:]
    X_norm = tl.norm(X, 2)
    rec_errors = []

    S, C, Y, Z = initialize(X, n_dim_c, n_dim_s, n_season, t)
    Es = X - Z  # residual tensor to update S
    Ec = X - Y  # residual tensor to update C

    for iteration in range(max_iter):
        print(f"Iter={iteration+1}")

        # Update Latent Interactions
        for mode in range(n_modes):
            if mode == 0:
                mttkrp = tl.unfolded_dot_khatri_rao(
                    Ec, (None, C), mode)
                mttkrp = tl.clip(mttkrp, a_min=epsilon, a_max=None)
                print(mttkrp.shape)

            else:
                pass

        else:
            pass

        if S is not None:
            # Update Seasonality
            for mode in range(n_modes):
                if mode == 0:
                    pass

                else:
                    pass

        # Reconstruction
        pass


        # Check convergence
        pass

    return A, B, C, S, Y, Z
            

