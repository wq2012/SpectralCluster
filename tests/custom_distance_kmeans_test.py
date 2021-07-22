import unittest
import numpy as np
from spectralcluster import custom_distance_kmeans
from spectralcluster import utils


class TestCustomDistanceKmeans(unittest.TestCase):
  """Tests for the run_kmeans function with the CustomKMeans class."""

  def setUp(self):
    super().setUp()
    pass

  def test_6by2_matrix_cosine_dist(self):
    matrix = np.array([
        [1.0, 0.0],
        [1.1, 0.1],
        [0.0, 1.0],
        [0.1, 1.0],
        [0.9, -0.1],
        [0.0, 1.2],
    ])

    labels = custom_distance_kmeans.run_kmeans(
        matrix, n_clusters=2, max_iter=300, custom_dist="cosine")
    labels = utils.enforce_ordered_labels(labels)
    expected = np.array([0, 0, 1, 1, 0, 1])
    self.assertTrue(np.array_equal(expected, labels))

  def test_6by2_matrix_euclidean_dist(self):
    matrix = np.array([
        [1.0, 0.0],
        [1.1, 0.1],
        [0.0, 1.0],
        [0.1, 1.0],
        [0.9, -0.1],
        [0.0, 1.2],
    ])

    labels = custom_distance_kmeans.run_kmeans(
        matrix, n_clusters=2, max_iter=300, custom_dist="euclidean")
    labels = utils.enforce_ordered_labels(labels)
    expected = np.array([0, 0, 1, 1, 0, 1])
    self.assertTrue(np.array_equal(expected, labels))

  def test_1000by6_matrix_cosine_dist(self):
    matrix = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 400 +
                      [[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]] * 300 +
                      [[0.0, 0.0, 2.0, 0.0, 0.0, 0.0]] * 200 +
                      [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]] * 100)
    noisy = np.random.rand(1000, 6) * 2 - 1
    matrix = matrix + noisy * 0.1

    labels = custom_distance_kmeans.run_kmeans(
        matrix, n_clusters=4, max_iter=300, custom_dist="cosine")
    labels = utils.enforce_ordered_labels(labels)
    expected = np.array([0] * 400 + [1] * 300 + [2] * 200 + [3] * 100)
    self.assertTrue(np.array_equal(expected, labels))

  def test_1000by6_matrix_euclidean_dist(self):
    matrix = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 400 +
                      [[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]] * 300 +
                      [[0.0, 0.0, 2.0, 0.0, 0.0, 0.0]] * 200 +
                      [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]] * 100)
    noisy = np.random.rand(1000, 6) * 2 - 1
    matrix = matrix + noisy * 0.1

    labels = custom_distance_kmeans.run_kmeans(
        matrix, n_clusters=4, max_iter=300, custom_dist="euclidean")
    labels = utils.enforce_ordered_labels(labels)
    expected = np.array([0] * 400 + [1] * 300 + [2] * 200 + [3] * 100)
    self.assertTrue(np.array_equal(expected, labels))


if __name__ == "__main__":
  unittest.main()
