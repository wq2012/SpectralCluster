import numpy as np
import typing


class NaiveCentroid:
  """A cluster centroid of the Naive clustering algorithm."""

  def __init__(self, embedding: np.ndarray):
    """Create a new centroid."""
    self.embedding = embedding
    self.count = 1

  def merge(self, embedding: np.ndarray):
    """Merge a new embedding into the centroid."""
    self.embedding = (self.embedding * self.count + embedding) / (
        self.count + 1)
    self.count += 1

  def cosine(self, embedding: np.ndarray) -> float:
    """Compute cosine similarity to a new embedding."""
    return np.dot(self.embedding, embedding) / (
        np.linalg.norm(self.embedding) * np.linalg.norm(embedding))


class NaiveClusterer:
  """Naive clustering class."""

  def __init__(self,
               threshold: float,
               adaptation_threshold: typing.Optional[float] = None):
    """Initialized the clusterer.

    Note that since this is online clustering, fit_predict and predict
    are the same.

    Args:
      threshold: if cosine similarity is larger than this threshold, the
        embedding will be considered to belong to the cluster
      adaptation_threshold: if cosine similarity is larger than
        adaptation_threshold, the embedding will be merged to the cluster.
        If None, we use threshold as adaptation_threshold

    Raises:
      ValueError: if adaptation_threshold is smaller than threshold
    """
    self.threshold = threshold
    if adaptation_threshold is None:
      self.adaptation_threshold = threshold
    elif adaptation_threshold < threshold:
      raise ValueError("adaptation_threshold cannot be smaller than threshold")
    else:
      self.adaptation_threshold = adaptation_threshold
    self.centroids = []

  def reset(self):
    """Reset the clusterer."""
    self.centroids = []

  def predict_next(self, embedding: np.ndarray) -> int:
    """Given a new embedding, output its label.

    This is used for online clustering.

    Args:
      embedding: numpy array of shape (n_features,)

    Returns:
      label: an integer cluster label
    """
    # Handle first embedding case.
    if len(self.centroids) == 0:
      self.centroids.append(NaiveCentroid(embedding))
      return 0

    # Compute all similarities.
    similarities = np.array(
        [centroid.cosine(embedding) for centroid in self.centroids])

    # New cluster.
    if similarities.max() < self.threshold:
      self.centroids.append(NaiveCentroid(embedding))
      return len(self.centroids) - 1

    # Existing cluster.
    label = similarities.argmax()
    if similarities[label] > self.adaptation_threshold:
      self.centroids[label].merge(embedding)
    return label

  def predict(self, embeddings: np.ndarray) -> np.ndarray:
    """Given many embeddings, return all cluster labels.

    This is for simulating offline clustering behavior.

    Args:
      embeddings: numpy array of shape (n_samples, n_features)

    Returns:
      labels: numpy array of shape (n_samples,)
    """
    return np.array([self.predict_next(embedding) for embedding in embeddings])

  def fit_predict(self, embeddings: np.ndarray) -> np.ndarray:
    """Same as predict(), since this is an online clusterer."""
    return self.predict(embeddings)
