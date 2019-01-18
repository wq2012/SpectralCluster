import abc
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


class Diffuse(AffinityRefinementOperation):
    """The diffusion operation."""
    def refine(self, X):
        self.check_input(X)
        return np.matmul(X, np.transpose(X))
