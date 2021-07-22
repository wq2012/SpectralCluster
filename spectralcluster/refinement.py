"""Affinity matrix refinemnet operations."""

import abc
import numpy as np
from scipy.ndimage import gaussian_filter

DEFAULT_REFINEMENT_SEQUENCE = [
    "CropDiagonal",
    "GaussianBlur",
    "RowWiseThreshold",
    "Symmetrize",
    "Diffuse",
    "RowWiseNormalize",
]


class RefinementOptions(object):
  """Refinement options for the affinity matrix."""

  def __init__(self,
               gaussian_blur_sigma=1,
               p_percentile=0.95,
               thresholding_soft_multiplier=0.01,
               thresholding_with_row_max=True,
               refinement_sequence=DEFAULT_REFINEMENT_SEQUENCE):
    """Initialization of the refinement arguments.

    Args:
      gaussian_blur_sigma: sigma value of the Gaussian blur operation
      p_percentile: the p-percentile for the row wise thresholding
      thresholding_soft_multiplier: the multiplier for soft threhsold, if this
        value is 0, then it's a hard thresholding
      thresholding_with_row_max: if true, we use row_max * p_percentile as row
        wise threshold, instead of doing a percentile-based thresholding
      refinement_sequence: a list of strings for the sequence of refinement
        operations to apply on the affinity matrix
    """
    self.gaussian_blur_sigma = gaussian_blur_sigma
    self.p_percentile = p_percentile
    self.thresholding_soft_multiplier = thresholding_soft_multiplier
    self.thresholding_with_row_max = thresholding_with_row_max
    self.refinement_sequence = refinement_sequence


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
               thresholding_with_row_max=False):
    self.p_percentile = p_percentile
    self.multiplier = thresholding_soft_multiplier
    self.thresholding_with_row_max = thresholding_with_row_max

  def refine(self, affinity):
    self.check_input(affinity)
    refined_affinity = np.copy(affinity)

    if self.thresholding_with_row_max:
      # Row_max based thresholding
      row_max = refined_affinity.max(axis=1)
      row_max = np.expand_dims(row_max, axis=1)
      is_smaller = refined_affinity < (row_max * self.p_percentile)
    else:
      # Percentile based thresholding
      row_percentile = np.percentile(
          refined_affinity, self.p_percentile * 100, axis=1)
      row_percentile = np.expand_dims(row_percentile, axis=1)
      is_smaller = refined_affinity < row_percentile

    refined_affinity = (refined_affinity * np.invert(is_smaller)) + (
        refined_affinity * self.multiplier * is_smaller)
    return refined_affinity


class Symmetrize(AffinityRefinementOperation):
  """The Symmetrization operation."""

  def refine(self, affinity):
    self.check_input(affinity)
    return np.maximum(affinity, np.transpose(affinity))


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
