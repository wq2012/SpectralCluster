from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import unittest
from spectralcluster import utils


class TestComputeAffinityMatrix(unittest.TestCase):
    """Tests for the compute_affinity_matrix function."""

    def test_4by2_matrix(self):
        X = np.array([[3, 4], [-4, 3], [6, 8], [-3, -4]])
        affinity = utils.compute_affinity_matrix(X)
        expected = np.array(
            [[1,  0.5, 1,  0, ],
             [0.5, 1, 0.5, 0.5],
             [1, 0.5, 1, 0],
             [0, 0.5, 0, 1]])
        self.assertTrue(np.array_equal(expected, affinity))


class TestComputeSortedEigenvectors(unittest.TestCase):
    """Tests for the compute_sorted_eigenvectors function."""

    def test_3by2_matrix(self):
        X = np.array([[1, 2], [3, 4], [1, 3]])
        affinity = utils.compute_affinity_matrix(X)
        w, v = utils.compute_sorted_eigenvectors(affinity)
        self.assertEqual((3, ), w.shape)
        self.assertEqual((3, 3), v.shape)
        self.assertGreater(w[0], w[1])
        self.assertGreater(w[1], w[2])


class TestComputeNumberOfClusters(unittest.TestCase):
    """Tests for the compute_number_of_clusters function."""

    def test_5_values(self):
        eigenvalues = np.array([1.0, 0.9, 0.8, 0.2, 0.1])
        result = utils.compute_number_of_clusters(eigenvalues)
        self.assertEqual(3, result)

    def test_max_clusters(self):
        max_clusters = 2
        eigenvalues = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5])

        result_1 = utils.compute_number_of_clusters(eigenvalues)
        self.assertEqual(5, result_1)

        result_2 = utils.compute_number_of_clusters(
            eigenvalues, max_clusters=max_clusters)
        self.assertEqual(max_clusters, result_2)


if __name__ == "__main__":
    unittest.main()
