"""
Covariance Estimation Module

Computes sample covariance matrix
from received array data.
"""

import numpy as np


def compute_covariance(X):
    """
    Computes sample covariance matrix.

    Parameters:
        X : (N x snapshots) complex array data

    Returns:
        R : (N x N) covariance matrix
    """

    num_snapshots = X.shape[1]

    # Sample covariance
    R = (X @ X.conj().T) / num_snapshots

    return R