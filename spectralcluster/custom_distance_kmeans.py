"""Implement custom kmeans.

It supports any distance measure defined in scipy.spatial.distance.
"""

from dataclasses import dataclass
import numpy as np
from scipy.spatial import distance
from sklearn.cluster import KMeans
import typing


def run_kmeans(spectral_embeddings: np.ndarray,
               n_clusters: int,
               custom_dist: typing.Union[str, typing.Callable],
               max_iter: int) -> np.ndarray:
  """Run CustomKMeans with a custom distance measure support.

  Perform a custom kmeans clustering with any distance measure defined
  in scipy.spatial.distance.

  Args:
    spectral_embeddings: input spectracl embedding observations
    n_clusters: the number of clusters to form
    custom_dist: str or callable. custom distance measure for k-means. if a
      string, "cosine", "euclidean", "mahalanobis", or any other distance
      functions defined in scipy.spatial.distance can be used
    max_iter: the maximum number of iterations for the custom k-means

  Returns:
    labels: predicted clustering labels of all samples
  """
  if not custom_dist:  # Scikit-learn KMeans
    kmeans_clusterer = KMeans(
        n_clusters=n_clusters, init="k-means++", max_iter=300, random_state=0)
  else:
    # Initialization using the k-means++ method in Scikit-learn
    kmeans_clusterer = KMeans(
        n_clusters=n_clusters, init="k-means++", max_iter=1, random_state=0)
    kmeans_clusterer.fit(spectral_embeddings)
    centroids = kmeans_clusterer.cluster_centers_
    # Run the cusotom K-means
    kmeans_clusterer = CustomKMeans(
        n_clusters=n_clusters,
        centroids=centroids,
        max_iter=max_iter,
        custom_dist=custom_dist)

  labels = kmeans_clusterer.predict(spectral_embeddings)
  return labels


@dataclass
class CustomKMeans:
  """Class CustomKMeans performs KMeans clustering."""

  # The number of clusters to form.
  n_clusters: typing.Optional[int] = None

  # The cluster centroids. If given, initial centroids are set as
  # the input samples. If not, centroids are randomly initialized.
  centroids: typing.Optional[np.ndarray] = None

  # Maximum number of iterations of the k-means algorithm to run.
  max_iter: int = 10

  # The relative increment in the results before declaring convergence.
  tol: float = 0.001

  # Custom distance measure to use. If a string, "cosine", "euclidean",
  # "mahalanobis", or any other distance functions
  # defined in scipy.spatial.distance can be used.
  custom_dist: typing.Union[str, typing.Callable] = "cosine"

  def _init_centroids(self, embeddings: np.ndarray):
    """Compute the initial centroids."""

    n_samples = embeddings.shape[0]
    idx = np.random.choice(
        np.arange(n_samples), size=self.n_clusters, replace=False)
    self.centroids = embeddings[idx, :]

  def predict(self, embeddings: np.ndarray) -> np.ndarray:
    """Performs the clustering.

    Args:
      embeddings: the input observations to cluster

    Returns:
      labels: predicted clustering labels of all samples

    Raises:
      ValueError: if input observations have wrong shape
    """
    n_samples, n_features = embeddings.shape
    if self.max_iter <= 0:
      raise ValueError("Number of iterations should be a positive number,"
                       " got %d instead" % self.max_iter)
    if n_samples < self.n_clusters:
      raise ValueError("n_samples=%d should be >= n_clusters=%d" %
                       (n_samples, self.n_clusters))
    if self.centroids is None:
      self._init_centroids(embeddings)
    else:
      n_centroids, c_n_features = self.centroids.shape
      if n_centroids != self.n_clusters:
        raise ValueError("The shape of the initial centroids (%s)"
                         "does not match the number of clusters %d" %
                         (str(self.centroids.shape), self.n_clusters))
      if n_features != c_n_features:
        raise ValueError(
            "The number of features of the initial centroids %d"
            "does not match the number of features of the data %d." %
            (c_n_features, n_features))

    sample_ids = np.arange(n_samples)
    prev_mean_dist = 0
    for iter_idx in range(self.max_iter + 1):
      # Compute distances to all centroids and assign each sample a label
      # corresponding to the nearest centroid
      dist_to_all_centroids = distance.cdist(
          embeddings, self.centroids, metric=self.custom_dist)
      labels = dist_to_all_centroids.argmin(axis=1)
      distances = dist_to_all_centroids[sample_ids, labels]
      mean_distance = np.mean(distances)
      # If the difference between current mean_distance and previous
      # mean_distance is very small or the max iteration number is reached,
      # the clustering stops
      if mean_distance <= prev_mean_dist and mean_distance >= (
          1 - self.tol) * prev_mean_dist or iter_idx == self.max_iter:
        break
      prev_mean_dist = mean_distance
      # Update centroids
      for each_centroid_idx in range(n_centroids):
        each_centroid_samples = np.where(labels == each_centroid_idx)[0]
        if each_centroid_samples.any():
          self.centroids[each_centroid_idx] = np.mean(
              embeddings[each_centroid_samples], axis=0)
    return labels
