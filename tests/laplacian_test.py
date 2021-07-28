import unittest

import numpy as np
from spectralcluster import laplacian
from spectralcluster import utils

LaplacianType = laplacian.LaplacianType


class TestComputeLaplacian(unittest.TestCase):
  """Tests for the compute_laplacian function."""

  def test_affinity(self):
    matrix = np.array([[3, 4], [-4, 3], [6, 8], [-3, -4]])
    affinity = utils.compute_affinity_matrix(matrix)
    result = laplacian.compute_laplacian(
        affinity, laplacian_type=LaplacianType.Affinity)
    self.assertTrue(np.array_equal(affinity, result))

  def test_laplacian(self):
    matrix = np.array([[3, 4], [-4, 3], [6, 8], [-3, -4]])
    affinity = utils.compute_affinity_matrix(matrix)
    laplacian_matrix = laplacian.compute_laplacian(
        affinity, laplacian_type=LaplacianType.Unnormalized)
    expected = np.array([[1.5, -0.5, -1, 0], [-0.5, 1.5, -0.5, -0.5],
                         [-1, -0.5, 1.5, 0], [0, -0.5, 0, 0.5]])
    self.assertTrue(np.array_equal(expected, laplacian_matrix))

  def test_normalized_laplacian(self):
    matrix = np.array([[3, 4], [-4, 3], [6, 8], [-3, -4]])
    affinity = utils.compute_affinity_matrix(matrix)
    laplacian_norm = laplacian.compute_laplacian(
        affinity, laplacian_type=LaplacianType.GraphCut)
    expected = np.array([[0.6, -0.2, -0.4, 0], [-0.2, 0.6, -0.2, -0.26],
                         [-0.4, -0.2, 0.6, 0], [0, -0.26, 0, 0.33]])
    self.assertTrue(np.allclose(expected, laplacian_norm, atol=0.01))

  def test_random_walk_normalized_laplacian(self):
    matrix = np.array([[3, 4], [-4, 3], [6, 8], [-3, -4]])
    affinity = utils.compute_affinity_matrix(matrix)
    laplacian_norm = laplacian.compute_laplacian(
        affinity, laplacian_type=LaplacianType.RandomWalk)
    expected = np.array([[0.6, -0.2, -0.4, 0], [-0.2, 0.6, -0.2, -0.2],
                         [-0.4, -0.2, 0.6, 0], [0, -0.33, 0, 0.33]])
    self.assertTrue(np.allclose(expected, laplacian_norm, atol=0.01))


if __name__ == "__main__":
  unittest.main()
