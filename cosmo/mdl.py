""" Definition of Minimum Discription Length """

import numpy as np
from sklearn import preprocessing
from scipy.stats import norm


def encoding_score(X, Y, avoid_negative_score=True):
    """
        X: input data (matrix/tensor)
        Y: reconstruction of X
    """
    diff = (X - Y).ravel()
    prob = norm.pdf(diff, loc=diff.mean(), scale=diff.std())

    # if avoid_negative_score == True:
    #     prob[prob > 1] = 1.

    # print(-1 * np.log2(prob).sum())

    return -1 * np.log2(prob).sum()


def model_score(X, normalize=False, float_cost=32, tol=1e-3):
    """
        X: input data (matrix/tensor)
        tol: threshold for zero values
    """
    score = 0

    if X.ndim == 1:
        k = X.shape[0]
        # if sparse component

        # X_nonzero = np.logical_or(X < -tol, tol < X).sum()
        X_nonzero = np.count_nonzero(np.logical_or(X < tol, tol < X))
        score += X_nonzero * (np.log(k) + float_cost)
        score += np.log1p(X_nonzero)

        # if dense component
        # score += np.prod(X.shape) * float_cost

    elif X.ndim == 2:
        k, l = X.shape  # k: # of dimensions of observations
        # X_nonzero = np.logical_or(X < -tol, tol < X).sum()
        # X_nonzero = np.count_nonzero(np.logical_or(X < tol, tol < X))
        X_nonzero = np.count_nonzero(X > tol)
        print('Nonzero=', X_nonzero)

        if normalize == True:
            score += X_nonzero * (np.log(k) + np.log(l) + float_cost) / k
        else:
            score += X_nonzero * (np.log(k) + np.log(l) + float_cost)

        score += np.log1p(X_nonzero)

        # score += np.prod(X.shape) * float_cost

    return score
