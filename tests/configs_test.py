import unittest
import numpy as np
from spectralcluster import configs
from spectralcluster import utils


class Icassp2018Test(unittest.TestCase):
  """Tests for ICASSP 2018 configs."""

  def test_1000by6_matrix(self):
    matrix = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 400 +
                      [[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]] * 300 +
                      [[0.0, 0.0, 2.0, 0.0, 0.0, 0.0]] * 200 +
                      [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]] * 100)
    noisy = np.random.rand(1000, 6) * 2 - 1
    matrix = matrix + noisy * 0.1
    labels = configs.icassp2018_clusterer.predict(matrix)
    labels = utils.enforce_ordered_labels(labels)
    expected = np.array([0] * 400 + [1] * 300 + [2] * 200 + [3] * 100)
    self.assertTrue(np.array_equal(expected, labels))


if __name__ == "__main__":
  unittest.main()
