"""Implementation of multi-stage clustering.

Multi-stage clustering class is introduced in this paper:

* Quan Wang, Yiling Huang, Han Lu, Guanlong Zhao, Ignacio Lopez Moreno,
  "Highly efficient real-time streaming and fully on-device speaker
  diarization with multi-stage clustering." arXiv preprint
  arXiv:2210.13690 (2022).
"""

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from spectralcluster import fallback_clusterer
from spectralcluster import spectral_clusterer
from spectralcluster import utils


class MultiStageClusterer:
  """Multi-stage clustering class."""

  def __init__(
      self,
      main_clusterer: spectral_clusterer.SpectralClusterer,
      fallback_threshold: float = 0.5,
      L: int = 50,
      U1: int = 100,
      U2: int = 600,
  ):
    # Main clusterer.
    self.main = main_clusterer

    if self.main.max_spectral_size:
      raise ValueError(
        "Do not set max_spectral_size for SpectralClusterer when"
        "using MultiStageClusterer.")

    # Lower bound of main clusterer.
    self.main.fallback_options.spectral_min_embeddings = L

    # Upper bound of main clusterer.
    self.U1 = U1

    # Upper bound of pre-clusterer.
    self.U2 = U2

    # Threshold for fallback AHC clusterer.
    self.main.fallback_options.agglomerative_threshold = fallback_threshold

    # Other configs for fallback.
    self.main.fallback_options.single_cluster_condition = (
      fallback_clusterer.SingleClusterCondition.FallbackClusterer)
    self.main.fallback_options.fallback_clusterer_type = (
      fallback_clusterer.FallbackClustererType.Agglomerative)

    # Pre-clusterer.
    self.pre = AgglomerativeClustering(
        n_clusters=U1,
        metric="cosine",
        linkage="complete")

    # All cached centroids.
    self.cache = None

    # Number of clustered embeddings.
    self.num_embeddings = 0

    # Array of shape (n_samples,), mapping from original embedding to compressed
    # centroid.
    self.compression_labels = None

  def streaming_predict(
      self,
      embedding: np.ndarray
  ) -> np.ndarray:
    """A streaming prediction function.

    Note that this is not a simple online prediction class It not only
    predicts the label of the next input, but also makes corrections to
    previously predicted labels.
    """
    self.num_embeddings += 1

    # First input.
    if self.num_embeddings == 1:
      self.cache = embedding
      return np.array([0])

    self.cache = np.vstack([self.cache, embedding])

    # Using fallback or main clusterer only.
    if self.num_embeddings <= self.U1:
      return self.main.predict(self.cache)

    # Run pre-clusterer.
    if self.compression_labels is not None:
      self.compression_labels = np.append(
          self.compression_labels, max(self.compression_labels) + 1)
    pre_labels = self.pre.fit_predict(self.cache)
    pre_centroids = utils.get_cluster_centroids(self.cache, pre_labels)
    main_labels = self.main.predict(pre_centroids)

    final_labels = utils.chain_labels(
        self.compression_labels,
        utils.chain_labels(pre_labels, main_labels))

    # Dynamic compression.
    if self.cache.shape[0] == self.U2:
      self.cache = pre_centroids
      self.compression_labels = utils.chain_labels(
        self.compression_labels, pre_labels)

    return final_labels
