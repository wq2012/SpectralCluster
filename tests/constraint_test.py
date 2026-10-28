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
      constraint.ConstraintOptions(  # pytype: disable=wrong-arg-types
          constraint_name="invalid", apply_before_refinement=False)

    with self.assertRaises(ValueError):
      constraint.AffinityIntegration(  # pytype: disable=wrong-arg-types
          integration_type="invalid").adjust_affinity(
              affinity, constraint_matrix)

  def test_3by3_matrix_cannot_link_and_mixed(self):
    affinity = np.array([
        [1.0, 0.8, 0.4],
        [0.8, 1.0, 0.2],
        [0.4, 0.2, 1.0],
    ])
    # Must-Link (1, 2), Cannot-Link (0, 1), soft Cannot-Link (0, 2) = -0.5
    constraint_matrix = np.array([
        [0.0, -1.0, -0.5],
        [-1.0, 0.0, 1.0],
        [-0.5, 1.0, 0.0],
    ])
    adjusted_max = constraint.AffinityIntegration(
        integration_type=IntegrationType.Max).adjust_affinity(
            affinity, constraint_matrix)
    expected_max = np.array([
        [1.0, 0.0, 0.2],
        [0.0, 1.0, 1.0],
        [0.2, 1.0, 1.0],
    ])
    np.testing.assert_allclose(adjusted_max, expected_max, atol=1e-6)

  def test_3by3_matrix_smooth(self):
    affinity = np.array([
        [1.0, 0.8, 0.4],
        [0.8, 1.0, 0.2],
        [0.4, 0.2, 1.0],
    ])
    # Soft Must-Link (1, 2) = 0.5, hard Cannot-Link (0, 1) = -1.0,
    # soft Cannot-Link (0, 2) = -0.5
    constraint_matrix = np.array([
        [0.0, -1.0, -0.5],
        [-1.0, 0.0, 0.5],
        [-0.5, 0.5, 0.0],
    ])
    adjusted_smooth = constraint.AffinityIntegration(
        integration_type=IntegrationType.Smooth).adjust_affinity(
            affinity, constraint_matrix)
    expected_smooth = np.array([
        [1.0, 0.0, 0.2],
        [0.0, 1.0, 0.6],
        [0.2, 0.6, 1.0],
    ])
    np.testing.assert_allclose(adjusted_smooth, expected_smooth, atol=1e-6)

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

  def test_invalid_alpha(self):
    with self.assertRaises(ValueError):
      constraint.ConstraintPropagation(alpha=0.0)
    with self.assertRaises(ValueError):
      constraint.ConstraintPropagation(alpha=-0.2)
    with self.assertRaises(ValueError):
      constraint.ConstraintPropagation(alpha=1.0)
    with self.assertRaises(ValueError):
      constraint.ConstraintPropagation(alpha=1.5)

  def test_normalized_constraint_propagation_must_link_and_cannot_link(self):
    # 4 samples: (0, 1) are similar, (2, 3) are similar
    affinity = np.array([
        [1.0, 0.9, 0.2, 0.2],
        [0.9, 1.0, 0.2, 0.2],
        [0.2, 0.2, 1.0, 0.9],
        [0.2, 0.2, 0.9, 1.0],
    ])
    # Must-Link on (1, 2) should set adj[1, 2] = 1.0 and propagate boost to
    # indirect pairs (0, 2), (1, 3), and (0, 3)
    ml_matrix = np.zeros((4, 4))
    ml_matrix[1, 2] = 1.0
    ml_matrix[2, 1] = 1.0
    options_ml = constraint.ConstraintOptions(
        constraint_name=constraint.ConstraintName.
        NormalizedConstraintPropagation,
        apply_before_refinement=True,
        constraint_propagation_alpha=0.6)
    adj_ml = options_ml.constraint_operator.adjust_affinity(
        affinity, ml_matrix)
    self.assertAlmostEqual(adj_ml[1, 2], 1.0)
    self.assertGreater(adj_ml[0, 2], affinity[0, 2])
    self.assertGreater(adj_ml[1, 3], affinity[1, 3])
    self.assertGreater(adj_ml[0, 3], affinity[0, 3])
    np.testing.assert_allclose(np.diag(adj_ml), np.ones(4))
    self.assertTrue(np.all((adj_ml >= 0.0) & (adj_ml <= 1.0)))

    # Cannot-Link on (0, 2) should set adj[0, 2] = 0.0 and propagate
    # suppression to indirect pairs (1, 2), (0, 3), and (1, 3), while
    # preserving diagonal self-affinities at 1.0
    cl_matrix = np.zeros((4, 4))
    cl_matrix[0, 2] = -1.0
    cl_matrix[2, 0] = -1.0
    adj_cl = options_ml.constraint_operator.adjust_affinity(
        affinity, cl_matrix)
    self.assertAlmostEqual(adj_cl[0, 2], 0.0)
    self.assertLess(adj_cl[1, 2], affinity[1, 2])
    self.assertLess(adj_cl[0, 3], affinity[0, 3])
    self.assertLess(adj_cl[1, 3], affinity[1, 3])
    np.testing.assert_allclose(np.diag(adj_cl), np.ones(4))
    self.assertTrue(np.all((adj_cl >= 0.0) & (adj_cl <= 1.0)))

    # Zero constraint matrix leaves affinity unchanged
    adj_zero = options_ml.constraint_operator.adjust_affinity(
        affinity, np.zeros((4, 4)))
    np.testing.assert_allclose(adj_zero, affinity)


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
