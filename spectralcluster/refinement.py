from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
from scipy.ndimage import gaussian_filter
import numpy as np


class AffinityRefinementOperation(metaclass=abc.ABCMeta):
    def check_input(self, X):
        """Check the input to the refine() method.

        Args:
            X: the input to the refine() method

        Raises:
            TypeError: if X has wrong type
            ValueError: if X has wrong shape, etc.
        """
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy array")
        shape = X.shape
        if len(shape) != 2:
            raise ValueError("X must be 2-dimensional")
        if shape[0] != shape[1]:
            raise ValueError("X must be a square matrix")

    @abc.abstractmethod
    def refine(self, X):
        """Perform the refinement operation.

        Args:
            X: the affinity matrix, of size (n_samples, n_samples)

        Returns:
            a matrix of the same size as X
        """
        pass


class CropDiagonal(AffinityRefinementOperation):
    """Crop the diagonal.

    Replace diagonal element by the max non-diagonal value of row.
    After this operation, the matrix has similar properties to a standard
    Laplacian matrix.
    This also helps to avoid the bias during Gaussian blur and normalization.
    """

    def refine(self, X):
        self.check_input(X)
        Y = np.copy(X)
        np.fill_diagonal(Y, 0.0)
        di = np.diag_indices(Y.shape[0])
        Y[di] = Y.max(axis=1)
        return Y


class GaussianBlur(AffinityRefinementOperation):
    """Apply Gaussian blur."""

    def __init__(self, sigma=1):
        self.sigma = sigma

    def refine(self, X):
        self.check_input(X)
        return gaussian_filter(X, sigma=self.sigma)


class RowWiseThreshold(AffinityRefinementOperation):
    """Apply row wise thresholding."""

    def __init__(self,
                 p_percentile=0.95,
                 thresholding_soft_multiplier=0.01,
                 thresholding_with_row_max=False):
        self.p_percentile = p_percentile
        self.multiplier = thresholding_soft_multiplier
        self.thresholding_with_row_max = thresholding_with_row_max

    def refine(self, X):
        self.check_input(X)
        Y = np.copy(X)

        if self.thresholding_with_row_max:
            # row_max based thresholding
            row_max = Y.max(axis=1)
            row_max = np.expand_dims(row_max, axis=1)
            is_smaller = Y < (row_max * self.p_percentile)
        else:
            # percentile based thresholding
            row_percentile = np.percentile(Y, self.p_percentile * 100, axis=1)
            row_percentile = np.expand_dims(row_percentile, axis=1)
            is_smaller = Y < row_percentile

        Y = (Y * np.invert(is_smaller)) + (Y * self.multiplier * is_smaller)
        return Y


class Symmetrize(AffinityRefinementOperation):
    """The Symmetrization operation."""

    def refine(self, X):
        self.check_input(X)
        return np.maximum(X, np.transpose(X))


class Diffuse(AffinityRefinementOperation):
    """The diffusion operation."""

    def refine(self, X):
        self.check_input(X)
        return np.matmul(X, np.transpose(X))


class RowWiseNormalize(AffinityRefinementOperation):
    """The row wise max normalization operation."""

    def refine(self, X):
        self.check_input(X)
        Y = np.copy(X)
        row_max = Y.max(axis=1)
        Y /= np.expand_dims(row_max, axis=1)
        return Y
