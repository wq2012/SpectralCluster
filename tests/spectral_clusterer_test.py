import numpy as np
import unittest
from spectralcluster import spectral_clusterer


class TestComputeAffinityMatrix(unittest.TestCase):
    """Tests for the compute_affinity_matrix function."""

    def test_4by2_matrix(self):
        X = np.array([[3, 4], [-4, 3], [6, 8], [-3, -4]])
        affinity = spectral_clusterer.compute_affinity_matrix(X)
        expected = np.array(
            [[1,  0.5, 1,  0, ],
             [0.5, 1, 0.5, 0.5],
             [1, 0.5, 1, 0],
             [0, 0.5, 0, 1]])
        self.assertTrue(np.array_equal(expected, affinity))

if __name__ == "__main__":
    unittest.main()
