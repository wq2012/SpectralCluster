"""Affinity matrix refinemnet operations."""

import abc
import enum
import numpy as np
from scipy.ndimage import gaussian_filter


class RefinementName(enum.Enum):
  """The names of the refinement operations."""
  CropDiagonal = 1
  GaussianBlur = 2
  RowWiseThreshold = 3
  Symmetrize = 4
  Diffuse = 5
  RowWiseNormalize = 6


class ThresholdType(enum.Enum):
  """Different types of thresholding."""
  # We clear values that are smaller than row_max*p_percentile
  RowMax = 1

  # We clear (p_percentile*100)% smallest values of the entire row
  Percentile = 2


class SymmetrizeType(enum.Enum):
  """Different types of symmetrization operation."""
  # We use max(A, A^T)
  Max = 1

  # We use 1/2(A + A^T)
  Average = 2


class RefinementOptions:
  """Refinement options for the affinity matrix."""

  def __init__(self,
               gaussian_blur_sigma=1,
               p_percentile=0.95,
               thresholding_soft_multiplier=0.01,
               thresholding_type=ThresholdType.RowMax,
               thresholding_with_binarization=False,
               thresholding_preserve_diagonal=False,
               symmetrize_type=SymmetrizeType.Max,
               refinement_sequence=None):
    """Initialization of the refinement arguments.

    Args:
      gaussian_blur_sigma: sigma value of the Gaussian blur operation
      p_percentile: the p-percentile for the row wise thresholding
      thresholding_soft_multiplier: the multiplier for soft threhsold, if this
        value is 0, then it's a hard thresholding
      thresholding_type: the type of thresholding operation
      thresholding_with_binarization: if true, we set values larger than the
        threshold to 1
      thresholding_preserve_diagonal: if true, in the row wise thresholding
        operation, we firstly set diagonals of the affinity matrix to 0, and set
        the diagonals back to 1 in the end
      symmetrize_type: a SymmetrizeType
      refinement_sequence: a list of RefinementName for the sequence of
        refinement operations to apply on the affinity matrix. If None, we will
        not refine
    """
    self.gaussian_blur_sigma = gaussian_blur_sigma
    self.p_percentile = p_percentile
    self.thresholding_soft_multiplier = thresholding_soft_multiplier
    self.thresholding_type = thresholding_type
    self.thresholding_with_binarization = thresholding_with_binarization
    self.thresholding_preserve_diagonal = thresholding_preserve_diagonal
    self.symmetrize_type = symmetrize_type
    if refinement_sequence is None:
      self.refinement_sequence = []
    elif not isinstance(refinement_sequence, list):
      raise TypeError("refinement_sequence must be a list of RefinementName")
    else:
      self.refinement_sequence = refinement_sequence

  def get_refinement_operator(self, name):
    """Get the refinement operator for the affinity matrix.

    Args:
      name: a RefinementName

    Returns:
      object of the operator

    Raises:
      TypeError: if name is not a RefinementName
      ValueError: if name is an unknown refinement operation
    """
    if not isinstance(name, RefinementName):
      raise TypeError("name must be a RefinementName")
    elif name == RefinementName.CropDiagonal:
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


class AffinityRefinementOperation(metaclass=abc.ABCMeta):
  """Refinement of the affinity matrix."""

  def check_input(self, affinity):
    """Check the input to the refine() method.

    Args:
      affinity: the input affinity matrix.

    Raises:
      TypeError: if affinity has wrong type
      ValueError: if affinity has wrong shape, etc.
    """
    if not isinstance(affinity, np.ndarray):
      raise TypeError("affinity must be a numpy array")
    shape = affinity.shape
    if len(shape) != 2:
      raise ValueError("affinity must be 2-dimensional")
    if shape[0] != shape[1]:
      raise ValueError("affinity must be a square matrix")

  @abc.abstractmethod
  def refine(self, affinity):
    """An abstract method to perform the refinement operation.

    Args:
        affinity: the affinity matrix, of size (n_samples, n_samples)

    Returns:
        a matrix of the same size as affinity
    """
    pass


class CropDiagonal(AffinityRefinementOperation):
  """Crop the diagonal.

  Replace diagonal element by the max non-diagonal value of row.
  After this operation, the matrix has similar properties to a standard
  Laplacian matrix. This also helps to avoid the bias during Gaussian blur and
  normalization.
  """

  def refine(self, affinity):
    self.check_input(affinity)
    refined_affinity = np.copy(affinity)
    np.fill_diagonal(refined_affinity, 0.0)
    di = np.diag_indices(refined_affinity.shape[0])
    refined_affinity[di] = refined_affinity.max(axis=1)
    return refined_affinity


class GaussianBlur(AffinityRefinementOperation):
  """Apply Gaussian blur."""

  def __init__(self, sigma=1):
    self.sigma = sigma

  def refine(self, affinity):
    self.check_input(affinity)
    return gaussian_filter(affinity, sigma=self.sigma)


class RowWiseThreshold(AffinityRefinementOperation):
  """Apply row wise thresholding."""

  def __init__(self,
               p_percentile=0.95,
               thresholding_soft_multiplier=0.01,
               thresholding_type=ThresholdType.RowMax,
               thresholding_with_binarization=False,
               thresholding_preserve_diagonal=False):
    self.p_percentile = p_percentile
    self.multiplier = thresholding_soft_multiplier
    if not isinstance(thresholding_type, ThresholdType):
      raise TypeError("thresholding_type must be a ThresholdType")
    self.thresholding_type = thresholding_type
    self.thresholding_with_binarization = thresholding_with_binarization
    self.thresholding_preserve_diagonal = thresholding_preserve_diagonal

  def refine(self, affinity):
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

  def __init__(self, symmetrize_type=SymmetrizeType.Max):
    if not isinstance(symmetrize_type, SymmetrizeType):
      raise TypeError("symmetrize_type must be a SymmetrizeType")
    self.symmetrize_type = symmetrize_type

  def refine(self, affinity):
    self.check_input(affinity)
    if self.symmetrize_type == SymmetrizeType.Max:
      return np.maximum(affinity, np.transpose(affinity))
    elif self.symmetrize_type == SymmetrizeType.Average:
      return 0.5 * (affinity + np.transpose(affinity))
    else:
      raise ValueError("Unsupported symmetrize_type.")


class Diffuse(AffinityRefinementOperation):
  """The diffusion operation."""

  def refine(self, affinity):
    self.check_input(affinity)
    return np.matmul(affinity, np.transpose(affinity))


class RowWiseNormalize(AffinityRefinementOperation):
  """The row wise max normalization operation."""

  def refine(self, affinity):
    self.check_input(affinity)
    refined_affinity = np.copy(affinity)
    row_max = refined_affinity.max(axis=1)
    refined_affinity /= np.expand_dims(row_max, axis=1)
    return refined_affinity
