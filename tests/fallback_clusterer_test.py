import unittest
import numpy as np
from spectralcluster import fallback_clusterer
from spectralcluster import utils

FallbackOptions = fallback_clusterer.FallbackOptions
FallbackClusterer = fallback_clusterer.FallbackClusterer


class TestFallbackClusterer(unittest.TestCase):

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
    options = FallbackOptions()
    clusterer = FallbackClusterer(options)
    labels = clusterer.predict(matrix)
    labels = utils.enforce_ordered_labels(labels)
    expected = np.array([0, 0, 1, 1, 0, 1])
    self.assertTrue(np.array_equal(expected, labels))


if __name__ == "__main__":
  unittest.main()
