import unittest

import numpy as np
from spectralcluster import configs
from spectralcluster import constraint
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


class TurnToDiarizeTest(unittest.TestCase):
  """Tests for Turn-To-Diarize system configs."""

  def test_6by2_matrix(self):
    matrix = np.array([
        [1.0, 0.0],
        [1.1, 0.1],
        [0.0, 1.0],
        [0.1, 1.0],
        [0.9, -0.1],
        [0.0, 1.2],
    ])
    speaker_turn_scores = [0, 0, 1.5, 0, 1.5, 1.5]
    constraint_matrix = constraint.ConstraintMatrix(
        speaker_turn_scores, threshold=1).compute_diagonals()
    labels = configs.turntodiarize_clusterer.predict(matrix, constraint_matrix)
    labels = utils.enforce_ordered_labels(labels)
    expected = np.array([0, 0, 1, 1, 0, 1])
    self.assertTrue(np.array_equal(expected, labels))


if __name__ == "__main__":
  unittest.main()
