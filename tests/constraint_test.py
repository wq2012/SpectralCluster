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
    np.testing.assert_allclose(
        np.array(adjusted_affinity), np.array(expected), atol=0.01)

  def test_3by3_matrix_average_and_options(self):
    affinity = np.array([[1.0, 0.2, 0.0], [0.4, 1.0, 0.0], [0.0, 0.0, 1.0]])
    constraint_matrix = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]])
    adjusted_affinity = constraint.AffinityIntegration(
        integration_type=IntegrationType.Average).adjust_affinity(
            affinity, constraint_matrix)
    expected = np.array([[1.0, 0.6, 0.0], [0.7, 1.0, 0.0], [0.0, 0.0, 0.5]])
    np.testing.assert_allclose(adjusted_affinity, expected, atol=0.01)

    options = constraint.ConstraintOptions(
        constraint_name=constraint.ConstraintName.AffinityIntegration,
        apply_before_refinement=False)
    np.testing.assert_allclose(
        options.constraint_operator.adjust_affinity(
            affinity, constraint_matrix),
        np.maximum(affinity, constraint_matrix))

    with self.assertRaises(ValueError):
      constraint.ConstraintOptions(
          constraint_name="invalid", apply_before_refinement=False)

    with self.assertRaises(ValueError):
      constraint.AffinityIntegration(
          integration_type="invalid").adjust_affinity(
              affinity, constraint_matrix)

  def test_check_input_errors(self):
    op = constraint.AffinityIntegration()
    with self.assertRaises(ValueError):
      op.adjust_affinity(np.zeros((2,)), np.zeros((2, 2)))
    with self.assertRaises(ValueError):
      op.adjust_affinity(np.zeros((2, 3)), np.zeros((2, 2)))
    with self.assertRaises(ValueError):
      op.adjust_affinity(np.zeros((2, 2)), np.zeros((2,)))
    with self.assertRaises(ValueError):
      op.adjust_affinity(np.zeros((2, 2)), np.zeros((2, 3)))
    with self.assertRaises(ValueError):
      op.adjust_affinity(np.zeros((2, 2)), np.zeros((3, 3)))


class TestConstraintPropagation(unittest.TestCase):
  """Tests for the ConstraintPropagation class."""

  def test_3by3_matrix(self):
    affinity = np.array([[1, 0.25, 0], [0.31, 1, 0], [0, 0, 1]])
    constraint_matrix = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]])
    adjusted_affinity = constraint.ConstraintPropagation(
        alpha=0.6).adjust_affinity(affinity, constraint_matrix)
    expected = np.array([[1, 0.97, 0], [1.03, 1, 0], [0, 0, 1]])
    np.testing.assert_allclose(
        np.array(adjusted_affinity), np.array(expected), atol=0.01)


class TestConstraintMatrix(unittest.TestCase):
  """Tests for the ConstraintMatrix class."""

  def test_3by3_constraint_matrix(self):
    speaker_turn_scores = [0, 0, 14.308253288269043]
    constraint_matrix = constraint.ConstraintMatrix(
        speaker_turn_scores, threshold=1).compute_diagonals()
    expected = np.array([[0, 1, 0], [1, 0, -1], [0, -1, 0]])
    np.testing.assert_equal(constraint_matrix, expected)

  def test_3by3_constraint_matrix_threshold(self):
    speaker_turn_scores = [0, 0, 0.12095779925584793]
    constraint_matrix = constraint.ConstraintMatrix(
        speaker_turn_scores, threshold=1).compute_diagonals()
    expected = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
    np.testing.assert_equal(constraint_matrix, expected)

  def test_negative_speaker_turn_score(self):
    with self.assertRaises(ValueError):
      constraint.ConstraintMatrix([0, -0.5, 1.0])


if __name__ == "__main__":
  unittest.main()
