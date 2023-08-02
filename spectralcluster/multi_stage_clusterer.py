"""Implementation of multi-stage clustering.

Multi-stage clustering class is introduced in this paper:

* Quan Wang, Yiling Huang, Han Lu, Guanlong Zhao, Ignacio Lopez Moreno,
  "Highly efficient real-time streaming and fully on-device speaker
  diarization with multi-stage clustering." arXiv preprint
  arXiv:2210.13690 (2022).
"""

import enum
import numpy as np
from scipy import optimize
from sklearn.cluster import AgglomerativeClustering
from spectralcluster import fallback_clusterer
from spectralcluster import spectral_clusterer
from spectralcluster import utils


class Deflicker(enum.Enum):
  """Method to deflicker the streaming output labels."""
  # No deflicker.
  NoDeflicker = enum.auto()

  # Deflicker by enforcing order-based outputs.
  OrderBased = enum.auto()

  # Deflicker by matching previous output using Hungarian algorithm.
  Hungarian = enum.auto()


def match_labels(
    current: np.ndarray,
    previous: np.ndarray) -> np.ndarray:
  """Match current labels with previous labels using Hungarian algorithm."""
  # We can assign each label in current to one or many label(s) in previous.
  current = utils.enforce_ordered_labels(current).astype(np.int32)
  previous = previous.astype(np.int32)
  current_crop = current[:-1]
  if current_crop.shape != previous.shape:
    raise ValueError("current must have one more element than previous .")
  num_current = max(current_crop) + 1
  num_previous = max(max(previous) + 1, num_current)

  # Compute cost matrix.
  cost = np.zeros((num_current, num_previous), dtype=np.int32)
  for i in range(num_current):
    for j in range(num_previous):
      cost[i, j] = np.sum(np.logical_and(current_crop == i, previous == j))

  # Solve assignment problem.
  row_ind, col_ind = optimize.linear_sum_assignment(cost, maximize=True)

  # Map labels.
  label_map = {}
  for i, j in zip(row_ind, col_ind):
    label_map[i] = j

  new_labels = current.copy()
  for i in range(max(current) + 1):
    if i in label_map:
      new_labels[current == i] = label_map[i]

  return new_labels


class MultiStageClusterer:
  """Multi-stage clustering class."""

  def __init__(
      self,
      main_clusterer: spectral_clusterer.SpectralClusterer,
      fallback_threshold: float = 0.5,
      L: int = 50,
      U1: int = 100,
      U2: int = 600,
      deflicker: Deflicker = Deflicker.NoDeflicker
  ):
    self.deflicker = deflicker

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

    self.previous_output = None

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
      final_labels = np.array([0])
      self.previous_output = final_labels
      return final_labels

    self.cache = np.vstack([self.cache, embedding])

    # Using fallback or main clusterer only.
    if self.num_embeddings <= self.U1:
      final_labels = self.main.predict(self.cache)
      self.previous_output = final_labels
      return final_labels

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

    # Deflicker.
    if self.num_embeddings > 1:
      if self.deflicker == Deflicker.OrderBased:
        final_labels = utils.enforce_ordered_labels(
            final_labels)
      elif self.deflicker == Deflicker.Hungarian:
        final_labels = match_labels(
          final_labels, self.previous_output)

    self.previous_output = final_labels
    return final_labels
