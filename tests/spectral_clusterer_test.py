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


class TestComputeSortedEigenvectors(unittest.TestCase):

    def test_3by2_matrix(self):
        X = np.array([[1, 2], [3, 4], [1, 3]])
        affinity = spectral_clusterer.compute_affinity_matrix(X)
        w, v = spectral_clusterer.compute_sorted_eigenvectors(affinity)
        self.assertEqual((3, ), w.shape)
        self.assertEqual((3, 3), v.shape)
        self.assertGreater(w[0], w[1])
        self.assertGreater(w[1], w[2])


class TestComputeNumberOfClusters(unittest.TestCase):

    def test_5_values(self):
        eigenvalues = np.array([1.0, 0.9, 0.8, 0.2, 0.1])
        result = spectral_clusterer.compute_number_of_clusters(eigenvalues)
        self.assertEqual(3, result)


if __name__ == "__main__":
    unittest.main()
