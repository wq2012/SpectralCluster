"""Utility functions."""

import enum
import numpy as np
import typing

EPS = 1e-10


class EigenGapType(enum.Enum):
  """Different types of the eigengap computation."""
  # Eigengap is the ratio of two eigenvalues
  Ratio = enum.auto()

  # Eigengap is the subtraction of two eigenvalues, and it is normalized
  # by the maximum eigenvalue
  NormalizedDiff = enum.auto()


def compute_affinity_matrix(embeddings: np.ndarray) -> np.ndarray:
  """Compute the affinity matrix from data.

  Note that the range of affinity is [0, 1].

  Args:
    embeddings: numpy array of shape (n_samples, n_features)

  Returns:
    affinity: numpy array of shape (n_samples, n_samples)
  """
  # Normalize the data.
  l2_norms = np.linalg.norm(embeddings, axis=1)
  embeddings_normalized = embeddings / l2_norms[:, None]
  # Compute cosine similarities. Range is [-1,1].
  cosine_similarities = np.matmul(embeddings_normalized,
                                  np.transpose(embeddings_normalized))
  # Compute the affinity. Range is [0,1].
  # Note that this step is not mentioned in the paper!
  affinity = (cosine_similarities + 1.0) / 2.0

  return affinity


def compute_sorted_eigenvectors(
    input_matrix: np.ndarray,
    descend: bool = True) -> typing.Tuple[np.ndarray, np.ndarray]:
  """Sort eigenvectors by the real part of eigenvalues.

  Args:
    input_matrix: the matrix to perform eigen analysis with shape (M, M)
    descend: sort eigenvalues in a descending order. Default is True

  Returns:
    w: sorted eigenvalues of shape (M,)
    v: sorted eigenvectors, where v[;, i] corresponds to ith largest
       eigenvalue
  """
  # Eigen decomposition.
  eigenvalues, eigenvectors = np.linalg.eig(input_matrix)
  eigenvalues = eigenvalues.real
  eigenvectors = eigenvectors.real
  if descend:
    # Sort from largest to smallest.
    index_array = np.argsort(-eigenvalues)
  else:
    # Sort from smallest to largest.
    index_array = np.argsort(eigenvalues)
  # Re-order.
  w = eigenvalues[index_array]
  v = eigenvectors[:, index_array]
  return w, v


def compute_number_of_clusters(eigenvalues: np.ndarray,
                               max_clusters: typing.Optional[int] = None,
                               stop_eigenvalue: float = 1e-2,
                               eigengap_type: EigenGapType = EigenGapType.Ratio,
                               descend: bool = True,
                               eps: float = EPS) -> typing.Tuple[int, float]:
  """Compute number of clusters using EigenGap principle.

  Use maximum EigenGap principle to find the number of clusters.

  Args:
    eigenvalues: sorted eigenvalues of the affinity matrix
    max_clusters: max number of clusters allowed
    stop_eigenvalue: we do not look at eigen values smaller than this
    eigengap_type: the type of the eigengap computation
    descend: sort eigenvalues in a descending order. Default is True
    eps: a small value for numerial stability

  Returns:
    max_delta_index: number of clusters as an integer
    max_delta_norm: normalized maximum eigen gap
  """
  if not isinstance(eigengap_type, EigenGapType):
    raise TypeError("eigengap_type must be a EigenGapType")
  max_delta = 0
  max_delta_index = 0
  range_end = len(eigenvalues)
  if max_clusters and max_clusters + 1 < range_end:
    range_end = max_clusters + 1

  if not descend:
    # The first eigen value is always 0 in an ascending order
    for i in range(1, range_end - 1):
      if eigengap_type == EigenGapType.Ratio:
        delta = eigenvalues[i + 1] / (eigenvalues[i] + eps)
      elif eigengap_type == EigenGapType.NormalizedDiff:
        delta = (eigenvalues[i + 1] - eigenvalues[i]) / np.max(eigenvalues)
      else:
        raise ValueError("Unsupported eigengap_type")
      if delta > max_delta:
        max_delta = delta
        max_delta_index = i + 1  # Index i means i+1 clusters
  else:
    for i in range(1, range_end):
      if eigenvalues[i - 1] < stop_eigenvalue:
        break
      if eigengap_type == EigenGapType.Ratio:
        delta = eigenvalues[i - 1] / (eigenvalues[i] + eps)
      elif eigengap_type == EigenGapType.NormalizedDiff:
        delta = (eigenvalues[i - 1] - eigenvalues[i]) / np.max(eigenvalues)
      else:
        raise ValueError("Unsupported eigengap_type")
      if delta > max_delta:
        max_delta = delta
        max_delta_index = i

  return max_delta_index, max_delta


def enforce_ordered_labels(labels: np.ndarray) -> np.ndarray:
  """Transform the label sequence to an ordered form.

  This is the same type of label sequence used in the paper "Discriminative
  neural clustering for speaker diarisation". This makes the label sequence
  permutation invariant.

  Args:
    labels: an array of integers

  Returns:
    new_labels: an array of integers, where it starts with 0 and smaller
      labels always appear first
  """
  new_labels = labels.copy()
  max_label = -1
  label_map = {}
  for element in labels.tolist():
    if element not in label_map:
      max_label += 1
      label_map[element] = max_label
  for key in label_map:
    new_labels[labels == key] = label_map[key]
  return new_labels
