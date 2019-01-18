from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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


def compute_number_of_clusters(eigenvalues, stop_eigenvalue=1e-3):
    """Compute number of clusters using EigenGap principle.

    Args:
        eigenvalues: sorted eigenvalues of the affinity matrix
        stop_eigenvalue: we do not look at eigen values smaller than this

    Returns:
        number of clusters as an integer
    """
    max_delta = 0
    max_delta_index = 0
    for i in range(1, len(eigenvalues)):
        if eigenvalues[i - 1] < stop_eigenvalue:
            break
        delta = eigenvalues[i - 1] / eigenvalues[i]
        if delta > max_delta:
            max_delta = delta
            max_delta_index = i
    return max_delta_index


class SpectralClusterer(object):
    def __init__(
            self,
            min_clusters=None,
            max_clusters=None,
            gaussian_blur_sigma=1,
            p_percentile=0.95,
            thresholding_soft_multiplier=0.01):
        """Constructor of the clusterer.

        Args:
            min_clusters: minimal number of clusters allowed (only effective
                if not None)
            max_clusters: maximal number of clusters allowed (only effective
                if not None), can be used together with min_clusters to fix
                the number of clusters
            gaussian_blur_sigma: sigma value of the Gaussian blur operation
            p_percentile: the p-percentile for the row wise thresholding
            thresholding_soft_multiplier: the multiplier for soft threhsold,
                if this value is 0, then it's a hard thresholding
        """
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.gaussian_blur_sigma = gaussian_blur_sigma
        self.p_percentile = p_percentile
        self.thresholding_soft_multiplier = thresholding_soft_multiplier

    def cluster(self, X):
        """Perform spectral clustering on data X.

        Args:
            X: numpy array of shape (n_samples, n_features)

        Returns:
            labels: numpy array of shape (n_samples,)

        Raises:
            TypeError: if X has wrong type
            ValueError: if X has wrong shape
        """
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy array")
        if len(X.shape) != 2:
            raise ValueError("X must be 2-dimensional")
        #  Compute affinity matrix.
        affinity = compute_affinity_matrix(X)

        # Refinement opertions on the affinity matrix.
        affinity = refinement.CropDiagonal().refine(affinity)
        affinity = refinement.GaussianBlur(
            self.gaussian_blur_sigma).refine(affinity)
        affinity = refinement.RowWiseThreshold(
            self.p_percentile,
            self.thresholding_soft_multiplier).refine(affinity)
        affinity = refinement.Symmetrize().refine(affinity)
        affinity = refinement.Diffuse().refine(affinity)
        affinity = refinement.RowWiseNormalize().refine(affinity)

        # Perform eigen decomposion.
        (eigenvalues, eigenvectors) = compute_sorted_eigenvectors(affinity)
        # Get number of clusters.
        k = compute_number_of_clusters(eigenvalues)
        if self.min_clusters is not None:
            k = max(k, self.min_clusters)
        if self.max_clusters is not None:
            k = min(k, self.max_clusters)

        # Get spectral embeddings.
        spectral_embeddings = eigenvectors[:, :k]

        # Run K-Means++ on spectral embeddings.
        # Note: The correct way should be using a K-Means implementation
        # that supports customized distance measure such as cosine distance.
        # This implemention from scikit-learn does NOT, which is inconsistent
        # with the paper.
        kmeans_clusterer = KMeans(
            n_clusters=k,
            init="k-means++",
            max_iter=300,
            random_state=0)
        labels = kmeans_clusterer.fit_predict(spectral_embeddings)
        return labels
