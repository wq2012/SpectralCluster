import unittest
import numpy as np
from spectralcluster import custom_distance_kmeans
from spectralcluster import utils


class TestCustomDistanceKmeans(unittest.TestCase):
  """Tests for the run_kmeans function with the CustomKMeans class."""

  def setUp(self):
    super().setUp()
    np.random.seed(0)

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
    np.testing.assert_equal(expected, labels)

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
    np.testing.assert_equal(expected, labels)

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
    np.testing.assert_equal(expected, labels)

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
    np.testing.assert_equal(expected, labels)

  def test_6by2_matrix_default_kmeans(self):
    matrix = np.array([
        [1.0, 0.0],
        [1.1, 0.1],
        [0.0, 1.0],
        [0.1, 1.0],
        [0.9, -0.1],
        [0.0, 1.2],
    ])
    labels = custom_distance_kmeans.run_kmeans(
        matrix, n_clusters=2, max_iter=300, custom_dist="")
    labels = utils.enforce_ordered_labels(labels)
    expected = np.array([0, 0, 1, 1, 0, 1])
    np.testing.assert_equal(expected, labels)

  def test_custom_kmeans_init_centroids_and_errors(self):
    np.random.seed(0)
    matrix = np.array([
        [1.0, 0.0],
        [1.1, 0.1],
        [0.0, 1.0],
        [0.1, 1.0],
        [0.9, -0.1],
        [0.0, 1.2],
    ])
    clusterer = custom_distance_kmeans.CustomKMeans(n_clusters=2, max_iter=50)
    labels = clusterer.predict(matrix)
    self.assertEqual((6,), labels.shape)

    with self.assertRaises(ValueError):
      custom_distance_kmeans.CustomKMeans(
          n_clusters=2, max_iter=0).predict(matrix)

    with self.assertRaises(ValueError):
      custom_distance_kmeans.CustomKMeans(
          n_clusters=10, max_iter=10).predict(matrix)

    with self.assertRaises(ValueError):
      custom_distance_kmeans.CustomKMeans(
          n_clusters=2, centroids=np.zeros((3, 2))).predict(matrix)

    with self.assertRaises(ValueError):
      custom_distance_kmeans.CustomKMeans(
          n_clusters=2, centroids=np.zeros((2, 3))).predict(matrix)


if __name__ == "__main__":
  unittest.main()
