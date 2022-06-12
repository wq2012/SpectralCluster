import unittest
import numpy as np
from spectralcluster import naive_clusterer
from spectralcluster import utils

NaiveClusterer = naive_clusterer.NaiveClusterer


class TestNaiveClusterer(unittest.TestCase):

  def setUp(self):
    super().setUp()
    pass

  def test_6by2_matrix(self):
    matrix = np.array([
        [1.0, 0.0],
        [1.1, 0.1],
        [0.0, 1.0],
        [0.1, 1.0],
        [0.9, -0.1],
        [0.0, 1.2],
    ])
    clusterer = NaiveClusterer(threshold=0.5)
    labels = clusterer.predict(matrix)
    labels = utils.enforce_ordered_labels(labels)
    expected = np.array([0, 0, 1, 1, 0, 1])
    np.testing.assert_equal(expected, labels)

    label = clusterer.predict_next(np.array([1.2, -0.1]))
    self.assertEqual(0, label)

    label = clusterer.predict_next(np.array([-0.1, 0.8]))
    self.assertEqual(1, label)

    clusterer.reset()
    label = clusterer.predict_next(np.array([-0.1, 0.8]))
    self.assertEqual(0, label)

  def test_adaptation(self):
    clusterer = NaiveClusterer(threshold=0.5, adaptation_threshold=1.0)
    label = clusterer.predict_next(np.array([1.2, -0.1]))
    self.assertEqual(0, label)
    self.assertEqual(1, clusterer.centroids[0].count)

    # adaptation_threshold is too big, won't adapt.
    label = clusterer.predict_next(np.array([1.3, 0.2]))
    self.assertEqual(0, label)
    self.assertEqual(1, clusterer.centroids[0].count)

    # adaptation_threshold is small, will adapt.
    clusterer.adaptation_threshold = 0.5
    label = clusterer.predict_next(np.array([1.3, 0.2]))
    self.assertEqual(0, label)
    self.assertEqual(2, clusterer.centroids[0].count)


if __name__ == "__main__":
  unittest.main()
