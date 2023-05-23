"""Affinity matrix refinemnet operations."""

import abc
from dataclasses import dataclass
import enum
import numpy as np
from scipy.ndimage import gaussian_filter
import typing


class RefinementName(enum.Enum):
  """The names of the refinement operations."""
  CropDiagonal = enum.auto()
  GaussianBlur = enum.auto()
  RowWiseThreshold = enum.auto()
  Symmetrize = enum.auto()
  Diffuse = enum.auto()
  RowWiseNormalize = enum.auto()


class ThresholdType(enum.Enum):
  """Different types of thresholding."""
  # We clear values that are smaller than row_max*p_percentile
  RowMax = enum.auto()

  # We clear (p_percentile*100)% smallest values of the entire row
  Percentile = enum.auto()


class SymmetrizeType(enum.Enum):
  """Different types of symmetrization operation."""
  # We use max(A, A^T)
  Max = enum.auto()

  # We use 1/2(A + A^T)
  Average = enum.auto()


class AffinityRefinementOperation(metaclass=abc.ABCMeta):
  """Refinement of the affinity matrix."""

  def check_input(self, affinity: np.ndarray):
    """Check the input to the refine() method.

    Args:
      affinity: the input affinity matrix.

    Raises:
      TypeError: if affinity has wrong type
      ValueError: if affinity has wrong shape, etc.
    """
    shape = affinity.shape
    if len(shape) != 2:
      raise ValueError("affinity must be 2-dimensional")
    if shape[0] != shape[1]:
      raise ValueError("affinity must be a square matrix")

  @abc.abstractmethod
  def refine(self, affinity: np.ndarray) -> np.ndarray:
    """An abstract method to perform the refinement operation.

    Args:
        affinity: the affinity matrix, of size (n_samples, n_samples)

    Returns:
        a matrix of the same size as affinity
    """
    pass


@dataclass
class RefinementOptions:
  """Refinement options for the affinity matrix."""

  # Sigma value of the Gaussian blur operation.
  gaussian_blur_sigma: int = 1

  # The p-percentile for the row wise thresholding.
  p_percentile: float = 0.95

  # The multiplier for soft threhsold, if this value is 0,
  # then it's a hard thresholding.
  thresholding_soft_multiplier: float = 0.01

  # The type of thresholding operation.
  thresholding_type: ThresholdType = ThresholdType.RowMax

  # If true, we set values larger than the threshold to 1.
  thresholding_with_binarization: bool = False

  # If true, in the row wise thresholding operation, we firstly set diagonals
  # of the affinity matrix to 0, and set the diagonals back to 1 in the end.
  thresholding_preserve_diagonal: bool = False

  # A SymmetrizeType.
  symmetrize_type: SymmetrizeType = SymmetrizeType.Max

  # A list of RefinementName for the sequence of refinement operations to
  # apply on the affinity matrix.
  refinement_sequence: typing.Optional[typing.Sequence[RefinementName]] = None

  def get_refinement_operator(self, name: RefinementName) -> (
      AffinityRefinementOperation):
    """Get the refinement operator for the affinity matrix.

    Args:
      name: a RefinementName

    Returns:
      object of the operator

    Raises:
      TypeError: if name is not a RefinementName
      ValueError: if name is an unknown refinement operation
    """
    if name == RefinementName.CropDiagonal:
      return CropDiagonal()
    elif name == RefinementName.GaussianBlur:
      return GaussianBlur(self.gaussian_blur_sigma)
    elif name == RefinementName.RowWiseThreshold:
      return RowWiseThreshold(self.p_percentile,
                              self.thresholding_soft_multiplier,
                              self.thresholding_type,
                              self.thresholding_with_binarization,
                              self.thresholding_preserve_diagonal)
    elif name == RefinementName.Symmetrize:
      return Symmetrize(self.symmetrize_type)
    elif name == RefinementName.Diffuse:
      return Diffuse()
    elif name == RefinementName.RowWiseNormalize:
      return RowWiseNormalize()
    else:
      raise ValueError("Unknown refinement operation: {}".format(name))


class CropDiagonal(AffinityRefinementOperation):
  """Crop the diagonal.

  Replace diagonal element by the max non-diagonal value of row.
  After this operation, the matrix has similar properties to a standard
  Laplacian matrix. This also helps to avoid the bias during Gaussian blur and
  normalization.
  """

  def refine(self, affinity: np.ndarray) -> np.ndarray:
    self.check_input(affinity)
    refined_affinity = np.copy(affinity)
    np.fill_diagonal(refined_affinity, 0.0)
    di = np.diag_indices(refined_affinity.shape[0])
    refined_affinity[di] = refined_affinity.max(axis=1)
    return refined_affinity


class GaussianBlur(AffinityRefinementOperation):
  """Apply Gaussian blur."""

  def __init__(self, sigma: int = 1):
    self.sigma = sigma

  def refine(self, affinity: np.ndarray) -> np.ndarray:
    self.check_input(affinity)
    return gaussian_filter(affinity, sigma=self.sigma)


class RowWiseThreshold(AffinityRefinementOperation):
  """Apply row wise thresholding."""

  def __init__(self,
               p_percentile: float = 0.95,
               thresholding_soft_multiplier: float = 0.01,
               thresholding_type: ThresholdType = ThresholdType.RowMax,
               thresholding_with_binarization: bool = False,
               thresholding_preserve_diagonal: bool = False):
    self.p_percentile = p_percentile
    self.multiplier = thresholding_soft_multiplier
    if not isinstance(thresholding_type, ThresholdType):
      raise TypeError("thresholding_type must be a ThresholdType")
    self.thresholding_type = thresholding_type
    self.thresholding_with_binarization = thresholding_with_binarization
    self.thresholding_preserve_diagonal = thresholding_preserve_diagonal

  def refine(self, affinity: np.ndarray) -> np.ndarray:
    self.check_input(affinity)
    refined_affinity = np.copy(affinity)
    if self.thresholding_preserve_diagonal:
      np.fill_diagonal(refined_affinity, 0.0)
    if self.thresholding_type == ThresholdType.RowMax:
      # Row_max based thresholding
      row_max = refined_affinity.max(axis=1)
      row_max = np.expand_dims(row_max, axis=1)
      is_smaller = refined_affinity < (row_max * self.p_percentile)
    elif self.thresholding_type == ThresholdType.Percentile:
      # Percentile based thresholding
      row_percentile = np.percentile(
          refined_affinity, self.p_percentile * 100, axis=1)
      row_percentile = np.expand_dims(row_percentile, axis=1)
      is_smaller = refined_affinity < row_percentile
    else:
      raise ValueError("Unsupported thresholding_type")
    if self.thresholding_with_binarization:
      # For values larger than the threshold, we binarize them to 1
      refined_affinity = (np.ones_like(
          (refined_affinity)) * np.invert(is_smaller)) + (
              refined_affinity * self.multiplier * is_smaller)
    else:
      refined_affinity = (refined_affinity * np.invert(is_smaller)) + (
          refined_affinity * self.multiplier * is_smaller)
    if self.thresholding_preserve_diagonal:
      np.fill_diagonal(refined_affinity, 1.0)
    return refined_affinity


class Symmetrize(AffinityRefinementOperation):
  """The Symmetrization operation."""

  def __init__(self, symmetrize_type: SymmetrizeType = SymmetrizeType.Max):
    self.symmetrize_type = symmetrize_type

  def refine(self, affinity: np.ndarray) -> np.ndarray:
    self.check_input(affinity)
    if self.symmetrize_type == SymmetrizeType.Max:
      return np.maximum(affinity, np.transpose(affinity))
    elif self.symmetrize_type == SymmetrizeType.Average:
      return 0.5 * (affinity + np.transpose(affinity))
    else:
      raise ValueError("Unsupported symmetrize_type.")


class Diffuse(AffinityRefinementOperation):
  """The diffusion operation."""

  def refine(self, affinity: np.ndarray) -> np.ndarray:
    self.check_input(affinity)
    return np.matmul(affinity, np.transpose(affinity))


class RowWiseNormalize(AffinityRefinementOperation):
  """The row wise max normalization operation."""

  def refine(self, affinity: np.ndarray) -> np.ndarray:
    self.check_input(affinity)
    refined_affinity = np.copy(affinity)
    row_max = refined_affinity.max(axis=1)
    refined_affinity /= np.expand_dims(row_max, axis=1)
    return refined_affinity
