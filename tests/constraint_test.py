import unittest
import numpy as np
from spectralcluster import constraint

IntegrationType = constraint.IntegrationType


class TestAffinityIntegration(unittest.TestCase):
  """Tests for the AffinityIntegration class."""

  def test_3by3_matrix(self):
    affinity = np.array([[1, 0.25, 0], [0.31, 1, 0], [0, 0, 1]])
    constraint_matrix = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]])
    adjusted_affinity = constraint.AffinityIntegration(
        integration_type=IntegrationType.Max).adjust_affinity(
            affinity, constraint_matrix)
    expected = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1]])
    self.assertTrue(
        np.allclose(np.array(adjusted_affinity), np.array(expected), atol=0.01))


class TestConstraintPropagation(unittest.TestCase):
  """Tests for the ConstraintPropagation class."""

  def test_3by3_matrix(self):
    affinity = np.array([[1, 0.25, 0], [0.31, 1, 0], [0, 0, 1]])
    constraint_matrix = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]])
    adjusted_affinity = constraint.ConstraintPropagation(
        alpha=0.6).adjust_affinity(affinity, constraint_matrix)
    expected = np.array([[1, 0.97, 0], [1.03, 1, 0], [0, 0, 1]])
    self.assertTrue(
        np.allclose(np.array(adjusted_affinity), np.array(expected), atol=0.01))


if __name__ == "__main__":
  unittest.main()
