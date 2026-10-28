"""Laplacian matrix."""

import enum
import numpy as np

EPS = 1e-10


class LaplacianType(enum.Enum):
  """Different types of Laplacian matrix."""
  # The affinity matrix, not a Laplacian: W
  Affinity = enum.auto()

  # The unnormalized Laplacian: L = D - W
  Unnormalized = enum.auto()

  # The random walk view normalized Laplacian:  D^{-1} * L
  RandomWalk = enum.auto()

  # The graph cut view normalized Laplacian: D^{-1/2} * L * D^{-1/2}
  GraphCut = enum.auto()


def compute_laplacian(affinity: np.ndarray,
                      laplacian_type: LaplacianType = LaplacianType.GraphCut,
                      eps: float = EPS) -> np.ndarray:
  """Compute the Laplacian matrix.

  Args:
    affinity: the affinity matrix of input data
    laplacian_type: a LaplacianType
    eps: a small value for numerical stability

  Returns:
    the Laplacian matrix

  Raises:
    TypeError: if laplacian_type is not a LaplacianType
    ValueError: if laplacian_type is not supported
  """
  degree = np.sum(affinity, axis=1)
  laplacian = np.diag(degree) - affinity
  if not isinstance(laplacian_type, LaplacianType):
    raise TypeError("laplacian_type must be a LaplacianType")
  elif laplacian_type == LaplacianType.Affinity:
    return affinity
  elif laplacian_type == LaplacianType.Unnormalized:
    return laplacian
  elif laplacian_type == LaplacianType.RandomWalk:
    # Random walk normalized version
    degree_norm = 1.0 / (degree + eps)
    return degree_norm[:, None] * laplacian
  elif laplacian_type == LaplacianType.GraphCut:
    # Graph cut normalized version
    degree_norm = 1.0 / (np.sqrt(degree) + eps)
    return degree_norm[:, None] * laplacian * degree_norm[None, :]
  else:
    raise ValueError("Unsupported laplacian_type.")
