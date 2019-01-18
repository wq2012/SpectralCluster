import numpy as np
from sklearn.cluster import KMeans
from spectralcluster import refinement


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


class SpectralClusterer(object):
    def __init__(self):
        pass

    def cluster(self, X):
        """Perform spectral clustering on data X.

        Args:
            X: numpy array of shape (n_samples, n_features)

        Returns:
            labels: numpy array of shape (n_samples,)
        """
        affinity = compute_affinity_matrix(X)
