import abc
import numpy as np


class AffinityRefinementOperation(metaclass=abc.ABCMeta):
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
        return np.matmul(X, np.transpose(X))
