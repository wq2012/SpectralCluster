"""A spectral clusterer class to perform clustering."""

import numpy as np
from spectralcluster import base_spectral_clusterer
from spectralcluster import custom_distance_kmeans
from spectralcluster import laplacian
from spectralcluster import refinement
from spectralcluster import utils

RefinementName = refinement.RefinementName


class SpectralClusterer(base_spectral_clusterer.BaseSpectralClusterer):
  """Spectral clustering class."""

  def __init__(self,
               min_clusters=None,
               max_clusters=None,
               refinement_options=None,
               autotune=None,
               laplacian_type=None,
               stop_eigenvalue=1e-2,
               row_wise_renorm=False,
               custom_dist="cosine",
               max_iter=300):
    """Constructor of the clusterer.

    Args:
      min_clusters: minimal number of clusters allowed (only effective if not
        None)
      max_clusters: maximal number of clusters allowed (only effective if not
        None), can be used together with min_clusters to fix the number of
        clusters
      refinement_options: a RefinementOptions object that contains refinement
        arguments for the affinity matrix
      autotune: an AutoTune object to automatically search p_percentile
      laplacian_type: a LaplacianType. If None, we do not use a laplacian matrix
      stop_eigenvalue: when computing the number of clusters using Eigen Gap, we
        do not look at eigen values smaller than this value
      row_wise_renorm: if True, perform row-wise re-normalization on the
        spectral embeddings
      custom_dist: str or callable. custom distance measure for k-means. if a
        string, "cosine", "euclidean", "mahalanobis", or any other distance
        functions defined in scipy.spatial.distance can be used
      max_iter: the maximum number of iterations for the custom k-means
    """
    super().__init__(
        min_clusters=min_clusters,
        max_clusters=max_clusters,
        refinement_options=refinement_options,
        autotune=autotune,
        laplacian_type=laplacian_type,
        stop_eigenvalue=stop_eigenvalue,
        row_wise_renorm=row_wise_renorm,
        custom_dist=custom_dist,
        max_iter=max_iter)

  def _compute_eigenvectors_ncluster(self, affinity):
    """Perform eigen decomposition and estiamte the number of clusters.

    Perform affinity refinement, eigen decomposition and sort eigenvectors by
    the real part of eigenvalues. Estimate the number of clusters using EigenGap
    principle.

    Args:
      affinity: the affinity matrix of input data

    Returns:
      eigenvectors: sorted eigenvectors. numpy array of shape
      (n_samples, n_samples)
      n_clusters: number of clusters as an integer
      max_delta_norm: normalized maximum eigen gap
    """
    # Perform refinement operations on the affinity matrix.
    for refinement_name in self.refinement_options.refinement_sequence:
      op = self._get_refinement_operator(refinement_name)
      affinity = op.refine(affinity)

    if not self.laplacian_type:
      # Perform eigen decomposion.
      (eigenvalues, eigenvectors) = utils.compute_sorted_eigenvectors(affinity)
      # Get number of clusters.
      n_clusters, max_delta_norm = utils.compute_number_of_clusters(
          eigenvalues, self.max_clusters, self.stop_eigenvalue, descend=True)
    else:
      # Compute Laplacian matrix
      laplacian_norm = laplacian.compute_laplacian(
          affinity, laplacian_type=self.laplacian_type)
      # Perform eigen decomposion. Eigen values are sorted in an ascending
      # order
      (eigenvalues, eigenvectors) = utils.compute_sorted_eigenvectors(
          laplacian_norm, descend=False)
      # Get number of clusters. Eigen values are sorted in an ascending order
      n_clusters, max_delta_norm = utils.compute_number_of_clusters(
          eigenvalues, self.max_clusters, descend=False)
    return eigenvectors, n_clusters, max_delta_norm

  def predict(self, embeddings):
    """Perform spectral clustering on data embeddings.

    The spectral clustering is performed on an affinity matrix.

    Args:
      embeddings: numpy array of shape (n_samples, n_features)

    Returns:
      labels: numpy array of shape (n_samples,)

    Raises:
      TypeError: if embeddings has wrong type
      ValueError: if embeddings has wrong shape
    """

    if not isinstance(embeddings, np.ndarray):
      raise TypeError("embeddings must be a numpy array")
    if len(embeddings.shape) != 2:
      raise ValueError("embeddings must be 2-dimensional")

    # Compute affinity matrix.
    affinity = utils.compute_affinity_matrix(embeddings)

    if self.autotune:
      # Use Auto-tuning method to find a good p_percentile.
      if (RefinementName.RowWiseThreshold
          not in self.refinement_options.refinement_sequence):
        raise ValueError(
            "AutoTune is only effective when the refinement sequence"
            "contains RowWiseThreshold")

      def p_percentile_to_ratio(p_percentile):
        """compute the `ratio` given a `p_percentile` value."""
        self.refinement_options.p_percentile = p_percentile
        (eigenvectors, n_clusters,
         max_delta_norm) = self._compute_eigenvectors_ncluster(affinity)
        ratio = (1 - p_percentile) / max_delta_norm
        return ratio, eigenvectors, n_clusters

      eigenvectors, n_clusters, _ = self.autotune.tune(p_percentile_to_ratio)
    else:
      # Do not use Auto-tune.
      eigenvectors, n_clusters, _ = self._compute_eigenvectors_ncluster(
          affinity)

    if self.min_clusters is not None:
      n_clusters = max(n_clusters, self.min_clusters)

    # Get spectral embeddings.
    spectral_embeddings = eigenvectors[:, :n_clusters]

    if self.row_wise_renorm:
      # Perfrom row wise re-normalization.
      rows_norm = np.linalg.norm(spectral_embeddings, axis=1, ord=2)
      spectral_embeddings = spectral_embeddings / np.reshape(
          rows_norm, (spectral_embeddings.shape[0], 1))

    # Run K-means on spectral embeddings.
    labels = custom_distance_kmeans.run_kmeans(
        spectral_embeddings,
        n_clusters=n_clusters,
        custom_dist=self.custom_dist,
        max_iter=self.max_iter)
    return labels
