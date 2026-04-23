"""
Distance / kernel functions used by from-scratch classifiers.
"""
import numpy as np


def euclidean_distance_sq(A, B):
    """
    Pairwise squared Euclidean distance between rows of A and B.
    Uses ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b for efficiency.
    Input:
        A: (n_a, d) array
        B: (n_b, d) array
    Return:
        D2: (n_a, n_b), D2[i, j] = ||A_i - B_j||^2
    """
    sqA = (A * A).sum(axis=1)[:, None]
    sqB = (B * B).sum(axis=1)[None, :]
    return np.maximum(sqA + sqB - 2.0 * A @ B.T, 0.0)


def rbf_kernel(A, B, gamma):
    """RBF kernel: K[i, j] = exp(-gamma * ||A_i - B_j||^2)."""
    return np.exp(-gamma * euclidean_distance_sq(A, B))
