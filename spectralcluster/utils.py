from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def compute_affinity_matrix(X):
    """Compute the affinity matrix from data.

    Note that the range of affinity is [0,1].

    Args:
        X: numpy array of shape (n_samples, n_features)

    Returns:
        affinity: numpy array of shape (n_samples, n_samples)
    """
    # Normalize the data.
    l2_norms = np.linalg.norm(X, axis=1)
    X_normalized = X / l2_norms[:, None]
    # Compute cosine similarities. Range is [-1,1].
    cosine_similarities = np.matmul(X_normalized, np.transpose(X_normalized))
    # Compute the affinity. Range is [0,1].
    # Note that this step is not mentioned in the paper!
    affinity = (cosine_similarities + 1.0) / 2.0
    return affinity


def compute_sorted_eigenvectors(A):
    """Sort eigenvectors by the real part of eigenvalues.

    Args:
        A: the matrix to perform eigen analysis with shape (M, M)

    Returns:
        w: sorted eigenvalues of shape (M,)
        v: sorted eigenvectors, where v[;, i] corresponds to ith largest
           eigenvalue
    """
    # Eigen decomposition.
    eigenvalues, eigenvectors = np.linalg.eig(A)
    eigenvalues = eigenvalues.real
    eigenvectors = eigenvectors.real
    # Sort from largest to smallest.
    index_array = np.argsort(-eigenvalues)
    # Re-order.
    w = eigenvalues[index_array]
    v = eigenvectors[:, index_array]
    return w, v


def compute_number_of_clusters(
        eigenvalues, max_clusters=None, stop_eigenvalue=1e-2):
    """Compute number of clusters using EigenGap principle.

    Args:
        eigenvalues: sorted eigenvalues of the affinity matrix
        max_clusters: max number of clusters allowed
        stop_eigenvalue: we do not look at eigen values smaller than this

    Returns:
        number of clusters as an integer
    """
    max_delta = 0
    max_delta_index = 0
    range_end = len(eigenvalues)
    if max_clusters and max_clusters + 1 < range_end:
        range_end = max_clusters + 1
    for i in range(1, range_end):
        if eigenvalues[i - 1] < stop_eigenvalue:
            break
        delta = eigenvalues[i - 1] / eigenvalues[i]
        if delta > max_delta:
            max_delta = delta
            max_delta_index = i
    return max_delta_index
