from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn.cluster import KMeans
from spectralcluster import refinement
from spectralcluster import utils


DEFAULT_REFINEMENT_SEQUENCE = [
    "CropDiagonal",
    "GaussianBlur",
    "RowWiseThreshold",
    "Symmetrize",
    "Diffuse",
    "RowWiseNormalize",
]


class SpectralClusterer(object):
    def __init__(
            self,
            min_clusters=None,
            max_clusters=None,
            gaussian_blur_sigma=1,
            p_percentile=0.95,
            thresholding_soft_multiplier=0.01,
            thresholding_with_row_max=True,
            stop_eigenvalue=1e-2,
            refinement_sequence=DEFAULT_REFINEMENT_SEQUENCE):
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
            thresholding_with_row_max: if true, we use row_max * p_percentile
                as row wise threshold, instead of doing a percentile-based
                thresholding
            stop_eigenvalue: when computing the number of clusters using
                Eigen Gap, we do not look at eigen values smaller than this
                value
            refinement_sequence: a list of strings for the sequence of
                refinement operations to apply on the affinity matrix
        """
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.gaussian_blur_sigma = gaussian_blur_sigma
        self.p_percentile = p_percentile
        self.thresholding_soft_multiplier = thresholding_soft_multiplier
        self.thresholding_with_row_max = thresholding_with_row_max
        self.stop_eigenvalue = stop_eigenvalue
        self.refinement_sequence = refinement_sequence

    def _get_refinement_operator(self, name):
        """Get the refinement operator.

        Args:
            name: operator class name as a string

        Returns:
            object of the operator

        Raises:
            ValueError: if name is an unknown refinement operation
        """
        if name == "CropDiagonal":
            return refinement.CropDiagonal()
        elif name == "GaussianBlur":
            return refinement.GaussianBlur(self.gaussian_blur_sigma)
        elif name == "RowWiseThreshold":
            return refinement.RowWiseThreshold(
                self.p_percentile,
                self.thresholding_soft_multiplier,
                self.thresholding_with_row_max)
        elif name == "Symmetrize":
            return refinement.Symmetrize()
        elif name == "Diffuse":
            return refinement.Diffuse()
        elif name == "RowWiseNormalize":
            return refinement.RowWiseNormalize()
        else:
            raise ValueError("Unknown refinement operation: {}".format(name))

    def predict(self, X):
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
        affinity = utils.compute_affinity_matrix(X)

        # Refinement opertions on the affinity matrix.
        for refinement_name in self.refinement_sequence:
            op = self._get_refinement_operator(refinement_name)
            affinity = op.refine(affinity)

        # Perform eigen decomposion.
        (eigenvalues, eigenvectors) = utils.compute_sorted_eigenvectors(
            affinity)
        # Get number of clusters.
        k = utils.compute_number_of_clusters(
            eigenvalues, self.max_clusters, self.stop_eigenvalue)
        if self.min_clusters is not None:
            k = max(k, self.min_clusters)

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
