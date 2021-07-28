import unittest

import numpy as np
from spectralcluster import utils


class TestComputeAffinityMatrix(unittest.TestCase):
  """Tests for the compute_affinity_matrix function."""

  def test_4by2_matrix(self):
    matrix = np.array([[3, 4], [-4, 3], [6, 8], [-3, -4]])
    affinity = utils.compute_affinity_matrix(matrix)
    expected = np.array([[1, 0.5, 1, 0], [0.5, 1, 0.5, 0.5], [1, 0.5, 1, 0],
                         [0, 0.5, 0, 1]])
    self.assertTrue(np.array_equal(expected, affinity))


class TestComputeSortedEigenvectors(unittest.TestCase):
  """Tests for the compute_sorted_eigenvectors function."""

  def test_3by2_matrix(self):
    matrix = np.array([[1, 2], [3, 4], [1, 3]])
    affinity = utils.compute_affinity_matrix(matrix)
    w, v = utils.compute_sorted_eigenvectors(affinity)
    self.assertEqual((3,), w.shape)
    self.assertEqual((3, 3), v.shape)
    self.assertGreater(w[0], w[1])
    self.assertGreater(w[1], w[2])

  def test_ascend(self):
    matrix = np.array([[1, 2], [3, 4], [1, 3]])
    affinity = utils.compute_affinity_matrix(matrix)
    w, v = utils.compute_sorted_eigenvectors(affinity, descend=False)
    self.assertEqual((3,), w.shape)
    self.assertEqual((3, 3), v.shape)
    self.assertLess(w[0], w[1])
    self.assertLess(w[1], w[2])


class TestComputeNumberOfClusters(unittest.TestCase):
  """Tests for the compute_number_of_clusters function."""

  def test_5_values(self):
    eigenvalues = np.array([1.0, 0.9, 0.8, 0.2, 0.1])
    result, max_delta_norm = utils.compute_number_of_clusters(eigenvalues)
    self.assertEqual(3, result)
    self.assertTrue(np.allclose(4.0, max_delta_norm, atol=0.01))

  def test_max_clusters(self):
    max_clusters = 2
    eigenvalues = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5])

    result_1, max_delta_norm_1 = utils.compute_number_of_clusters(eigenvalues)
    self.assertEqual(5, result_1)
    self.assertTrue(np.allclose(1.2, max_delta_norm_1, atol=0.01))

    result_2, max_delta_norm_2 = utils.compute_number_of_clusters(
        eigenvalues, max_clusters=max_clusters)
    self.assertEqual(max_clusters, result_2)
    self.assertTrue(np.allclose(1.125, max_delta_norm_2, atol=0.01))

  def test_ascend(self):
    eigenvalues = np.array([1.0, 0.9, 0.8, 0.2, 0.1])
    result, max_delta_norm = utils.compute_number_of_clusters(
        eigenvalues, max_clusters=3, descend=False)
    self.assertEqual(2, result)
    self.assertTrue(np.allclose(0.88, max_delta_norm, atol=0.01))


class TestEnforceOrderedLabels(unittest.TestCase):
  """Tests for the enforce_ordered_labels function."""

  def test_small_array(self):
    labels = np.array([2, 2, 1, 0, 3, 3, 1])
    expected = np.array([0, 0, 1, 2, 3, 3, 1])
    result = utils.enforce_ordered_labels(labels)
    self.assertTrue(np.array_equal(expected, result))


if __name__ == "__main__":
  unittest.main()
