"""Laplacian matrix."""

import enum
import numpy as np

EPS = 1e-10


class LaplacianType(enum.Enum):
  """Different types of Laplacian matrix."""
  # The affinity matrix, not a Laplacian: W
  Affinity = 0

  # The unnormalied Laplacian: L = D - W
  Unnormalized = 1

  # The random walk view normalized Laplacian:  D^{-1} * L
  RandomWalk = 2

  # The graph cut view normalized Laplacian: D^{-1/2} * L * D^{-1/2}
  GraphCut = 3


def compute_laplacian(affinity, laplacian_type=LaplacianType.GraphCut, eps=EPS):
  """Compute the Laplacian matrix.

  Args:
    affinity: the affinity matrix of input data
    laplacian_type: a LaplacianType
    eps: a small value for numerial stability

  Returns:
    the Laplacian matrix

  Raises:
    TypeError: if laplacian_type is not a LaplacianType
    ValueError: if laplacian_type is not supported
  """
  degree = np.diag(np.sum(affinity, axis=1))
  laplacian = degree - affinity
  if not isinstance(laplacian_type, LaplacianType):
    raise TypeError("laplacian_type must be a LaplacianType")
  elif laplacian_type == LaplacianType.Affinity:
    return affinity
  elif laplacian_type == LaplacianType.Unnormalized:
    return laplacian
  elif laplacian_type == LaplacianType.RandomWalk:
    # Random walk normalized version
    degree_norm = np.diag(1 / (np.diag(degree) + eps))
    laplacian_norm = degree_norm.dot(laplacian)
    return laplacian_norm
  elif laplacian_type == LaplacianType.GraphCut:
    # Graph cut normalized version
    degree_norm = np.diag(1 / (np.sqrt(np.diag(degree)) + eps))
    laplacian_norm = degree_norm.dot(laplacian).dot(degree_norm)
    return laplacian_norm
  else:
    raise ValueError("Unsupported laplacian_type.")
