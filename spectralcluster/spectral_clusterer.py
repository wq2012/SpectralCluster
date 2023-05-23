"""A spectral clusterer class to perform clustering."""

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from spectralcluster import autotune
from spectralcluster import constraint
from spectralcluster import custom_distance_kmeans
from spectralcluster import fallback_clusterer
from spectralcluster import laplacian
from spectralcluster import refinement
from spectralcluster import utils
import typing


AutoTune = autotune.AutoTune
AutoTuneProxy = autotune.AutoTuneProxy
ConstraintName = constraint.ConstraintName
ConstraintOptions = constraint.ConstraintOptions
FallbackOptions = fallback_clusterer.FallbackOptions
LaplacianType = laplacian.LaplacianType
RefinementName = refinement.RefinementName
RefinementOptions = refinement.RefinementOptions
EigenGapType = utils.EigenGapType


class SpectralClusterer:
  """Spectral clustering class."""

  def __init__(
      self,
      min_clusters: typing.Optional[int] = None,
      max_clusters: typing.Optional[int] = None,
      refinement_options: typing.Optional[RefinementOptions] = None,
      autotune: typing.Optional[AutoTune] = None,
      fallback_options: typing.Optional[FallbackOptions] = None,
      laplacian_type: typing.Optional[LaplacianType] = None,
      stop_eigenvalue: float = 1e-2,
      row_wise_renorm: bool = False,
      custom_dist: typing.Union[str, typing.Callable] = "cosine",
      max_iter: int = 300,
      constraint_options: typing.Optional[ConstraintOptions] = None,
      eigengap_type: EigenGapType = EigenGapType.Ratio,
      max_spectral_size: typing.Optional[int] = None,
      affinity_function: typing.Callable = utils.compute_affinity_matrix,
      post_eigen_cluster_function: typing.Callable = (
          custom_distance_kmeans.run_kmeans)):
    """Constructor of the clusterer.

    Args:
      min_clusters: minimal number of clusters allowed (only effective if not
        None)
      max_clusters: maximal number of clusters allowed (only effective if not
        None), can be used together with min_clusters to fix the number of
        clusters
      refinement_options: a RefinementOptions object that contains refinement
        arguments for the affinity matrix. If None, we will not refine
      autotune: an AutoTune object to automatically search p_percentile
      fallback_options: a FallbackOptions object to indicate when to run
        fallback clusterer instead of spectral clusterer
      laplacian_type: a LaplacianType. If None, we do not use a laplacian matrix
      stop_eigenvalue: when computing the number of clusters using Eigen Gap, we
        do not look at eigen values smaller than this value
      row_wise_renorm: if True, perform row-wise re-normalization on the
        spectral embeddings
      custom_dist: str or callable. custom distance measure for k-means. If a
        string, "cosine", "euclidean", "mahalanobis", or any other distance
        functions defined in scipy.spatial.distance can be used
      max_iter: the maximum number of iterations for the custom k-means
      constraint_options: a ConstraintOptions object that contains constraint
        arguments
      eigengap_type: the type of the eigengap computation
      max_spectral_size: the maximal size of input to the spectral clustering
        algorithm. If this is set, and the actual input size is larger than
        this value, then we are going to first use hierarchical clustering
        to reduce the input size to this number. This can significantly reduce
        the computational cost for steps like Laplacian matrix and eigen
        decomposition. However, please note that this may degrade the quality
        of the final clustering results. This corresponds to the U1 value in
        the multi-stage clustering paper (https://arxiv.org/abs/2210.13690)
      affinity_function: a function to compute the affinity matrix from the
        embeddings. This defaults to (cos(x,y)+1)/2
      post_eigen_cluster_function: a function to cluster the spectral embeddings
        after the eigenvalue computations. This function must have the same
        signature as custom_distance_kmeans.run_kmeans
    """
    self.min_clusters = min_clusters
    self.max_clusters = max_clusters
    if not refinement_options:
      self.refinement_options = refinement.RefinementOptions()
    else:
      self.refinement_options = refinement_options
    self.autotune = autotune
    if not fallback_options:
      self.fallback_options = fallback_clusterer.FallbackOptions()
    else:
      self.fallback_options = fallback_options
    self.laplacian_type = laplacian_type
    self.row_wise_renorm = row_wise_renorm
    self.stop_eigenvalue = stop_eigenvalue
    self.custom_dist = custom_dist
    self.max_iter = max_iter
    self.constraint_options = constraint_options
    self.eigengap_type = eigengap_type
    self.max_spectral_size = max_spectral_size
    self.affinity_function = affinity_function
    self.post_eigen_cluster_function = post_eigen_cluster_function

  def _compute_eigenvectors_ncluster(
      self,
      affinity: np.ndarray,
      constraint_matrix: typing.Optional[np.ndarray] = None) -> (
          typing.Tuple[np.ndarray, int, float]):
    """Perform eigen decomposition and estiamte the number of clusters.

    Perform affinity refinement, eigen decomposition and sort eigenvectors by
    the real part of eigenvalues. Estimate the number of clusters using EigenGap
    principle.

    Args:
      affinity: the affinity matrix of input data
      constraint_matrix: numpy array of shape (n_samples, n_samples). The
        constraint matrix with prior information

    Returns:
      eigenvectors: sorted eigenvectors. numpy array of shape
      (n_samples, n_samples)
      n_clusters: number of clusters as an integer
      max_delta_norm: normalized maximum eigen gap
    """
    # Perform refinement operations on the affinity matrix.
    if self.refinement_options.refinement_sequence:
      for refinement_name in self.refinement_options.refinement_sequence:
        refinement_operator = self.refinement_options.get_refinement_operator(
            refinement_name)
        affinity = refinement_operator.refine(affinity)

    if (self.constraint_options and
        not self.constraint_options.apply_before_refinement and
        constraint_matrix is not None):
      # Perform the constraint operation after refinement
      affinity = self.constraint_options.constraint_operator.adjust_affinity(
          affinity, constraint_matrix)

    if not self.laplacian_type or self.laplacian_type == LaplacianType.Affinity:
      # Perform eigen decomposion.
      (eigenvalues, eigenvectors) = utils.compute_sorted_eigenvectors(affinity)
      # Get number of clusters.
      n_clusters, max_delta_norm = utils.compute_number_of_clusters(
          eigenvalues,
          max_clusters=self.max_clusters,
          stop_eigenvalue=self.stop_eigenvalue,
          eigengap_type=self.eigengap_type,
          descend=True)
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
          eigenvalues,
          max_clusters=self.max_clusters,
          eigengap_type=self.eigengap_type,
          descend=False)
    return eigenvectors, n_clusters, max_delta_norm

  def _reduce_size_and_predict(self, embeddings: np.ndarray) -> np.ndarray:
    """Reduce the input size, then run spectral clustering.

    Args:
      embeddings: numpy array of shape (n_samples, n_features)

    Returns:
      labels: numpy array of shape (n_samples,)
    """
    # Run AHC on the input to reduce the size.
    # Note that linkage needs to be "complete", ecause "average" and "single"
    # do not work very well here.
    # Alternatively, we can use "euclidean" and "ward", but that requires
    # that the inputs are L2 normalized first.
    ahc = AgglomerativeClustering(
        n_clusters=self.max_spectral_size,
        affinity="cosine",
        linkage="complete")
    ahc_labels = ahc.fit_predict(embeddings)

    # Compute the centroids of the AHC clusters.
    ahc_centroids = []
    for i in range(self.max_spectral_size):
      ahc_cluster_embeddings = embeddings[ahc_labels == i, :]
      ahc_centroids.append(np.mean(ahc_cluster_embeddings, axis=0))
    ahc_centroids = np.stack(ahc_centroids)

    # Run spectral clustering on AHC centroids.
    spectral_labels = self.predict(ahc_centroids)

    # Convert spectral labels to final labels.
    final_labels = np.zeros(ahc_labels.shape)
    for i in range(self.max_spectral_size):
      final_labels[ahc_labels == i] = spectral_labels[i]

    return final_labels

  def predict(
      self,
      embeddings: np.ndarray,
      constraint_matrix: typing.Optional[np.ndarray] = None) -> np.ndarray:
    """Perform spectral clustering on data embeddings.

    The spectral clustering is performed on an affinity matrix.

    Args:
      embeddings: numpy array of shape (n_samples, n_features)
      constraint_matrix: numpy array of shape (n_samples, n_samples). The
        constraint matrix with prior information

    Returns:
      labels: numpy array of shape (n_samples,)

    Raises:
      TypeError: if embeddings has wrong type
      ValueError: if embeddings has wrong shape
      RuntimeError: if max_spectral_size is set and constraint_matrix is given
    """
    num_embeddings = embeddings.shape[0]

    if not isinstance(embeddings, np.ndarray):
      raise TypeError("embeddings must be a numpy array")
    if len(embeddings.shape) != 2:
      raise ValueError("embeddings must be 2-dimensional")

    # Check whether we need to run fallback clusterer instead.
    if (num_embeddings <
        self.fallback_options.spectral_min_embeddings):
      temp_clusterer = fallback_clusterer.FallbackClusterer(
          self.fallback_options)
      return temp_clusterer.predict(embeddings)

    # Check whether the input size is too big for running spectral clustering.
    if (self.max_spectral_size is not None
        and num_embeddings > self.max_spectral_size):
      if constraint_matrix is not None:
        raise RuntimeError(
            "Cannot handle constraint_matrix when max_spectral_size is set")
      if (self.max_spectral_size < 2 or
         (self.max_clusters and self.max_spectral_size <= self.max_clusters) or
         (self.min_clusters and self.max_spectral_size <= self.min_clusters)):
        raise ValueError(
            "max_spectral_size should be a relatively big number")
      return self._reduce_size_and_predict(embeddings)

    # Compute affinity matrix.
    affinity = self.affinity_function(embeddings)

    # Make single-vs-multi cluster(s) decision.
    if self.min_clusters == 1:
      if fallback_clusterer.check_single_cluster(
              self.fallback_options, embeddings, affinity):
        return np.array([0] * num_embeddings)

    # Apply constraint.
    if (self.constraint_options and
        self.constraint_options.apply_before_refinement and
        constraint_matrix is not None):
      # Perform the constraint operation before refinement
      affinity = self.constraint_options.constraint_operator.adjust_affinity(
          affinity, constraint_matrix)

    if self.autotune:
      # Use Auto-tuning method to find a good p_percentile.
      if (RefinementName.RowWiseThreshold
          not in self.refinement_options.refinement_sequence):
        raise ValueError(
            "AutoTune is only effective when the refinement sequence"
            "contains RowWiseThreshold")

      def p_percentile_to_ratio(p_percentile: float) -> (
          typing.Tuple[float, np.ndarray, int]):
        """Compute the `ratio` given a `p_percentile` value."""
        self.refinement_options.p_percentile = p_percentile
        (eigenvectors, n_clusters,
         max_delta_norm) = self._compute_eigenvectors_ncluster(
             affinity, constraint_matrix)
        if self.autotune.proxy == AutoTuneProxy.PercentileSqrtOverNME:
          ratio = np.sqrt(1 - p_percentile) / max_delta_norm
        elif self.autotune.proxy == AutoTuneProxy.PercentileSqrtOverNME:
          ratio = (1 - p_percentile) / max_delta_norm
        else:
          raise ValueError("Unsupported value of AutoTuneProxy")
        return ratio, eigenvectors, n_clusters

      eigenvectors, n_clusters, _ = self.autotune.tune(p_percentile_to_ratio)
    else:
      # Do not use Auto-tune.
      eigenvectors, n_clusters, _ = self._compute_eigenvectors_ncluster(
          affinity, constraint_matrix)

    if self.min_clusters is not None:
      n_clusters = max(n_clusters, self.min_clusters)

    # Get spectral embeddings.
    spectral_embeddings = eigenvectors[:, :n_clusters]

    if self.row_wise_renorm:
      # Perform row wise re-normalization.
      rows_norm = np.linalg.norm(spectral_embeddings, axis=1, ord=2)
      spectral_embeddings = spectral_embeddings / np.reshape(
          rows_norm, (num_embeddings, 1))

    # Run clustering algorithm on spectral embeddings. This defaults
    # to customized K-means.
    labels = self.post_eigen_cluster_function(
        spectral_embeddings=spectral_embeddings,
        n_clusters=n_clusters,
        custom_dist=self.custom_dist,
        max_iter=self.max_iter)
    return labels
