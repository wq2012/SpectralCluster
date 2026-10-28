"""Constraint information."""
import abc
from dataclasses import dataclass
import enum
import numpy as np
import typing

EPS = 1e-10


class ConstraintName(enum.Enum):
  """The names of constrained operations."""
  # The Affinity Integration method
  AffinityIntegration = enum.auto()

  # The Constraint Propagation method
  ConstraintPropagation = enum.auto()

  # Normalized Constraint Propagation (E2CP with zero-diagonal propagation
  # graph, normalized/clipped propagated constraints, and preserved diagonal)
  NormalizedConstraintPropagation = enum.auto()


class IntegrationType(enum.Enum):
  """The integration types for the Affinity Integration method."""
  # For non-negative constraints (Must-Link or unconstrained), take
  # np.maximum(affinity, constraint_matrix). For negative constraints
  # (Cannot-Link), scale affinity by (1 + constraint_matrix) * affinity.
  Max = enum.auto()

  # Take 0.5 * (affinity + constraint_matrix).
  Average = enum.auto()

  # Smooth adjustment from E2CP (Eq. 13 in Lu et al. 2013):
  # 1 - (1 - C) * (1 - A) for C >= 0, and (1 + C) * A for C < 0.
  Smooth = enum.auto()


@dataclass
class ConstraintOptions:
  """Constraint options for constrained clustering methods."""

  # Name of the constrained clustering method.
  constraint_name: ConstraintName

  # If True, this operation is applied before the affinity refinement.
  # It is suggested to set as True for the ConstraintPropagation method
  # and False for the AffinityIntegration method.
  apply_before_refinement: bool

  # Integration type for the Affinity Integration method.
  integration_type: typing.Optional[IntegrationType] = IntegrationType.Max

  # alpha value of the constraint propagation method.
  constraint_propagation_alpha: float = 0.6

  def __post_init__(self):
    if self.constraint_name == ConstraintName.AffinityIntegration:
      integration_type = self.integration_type or IntegrationType.Max
      self.constraint_operator = AffinityIntegration(integration_type)
    elif self.constraint_name == ConstraintName.ConstraintPropagation:
      self.constraint_operator = ConstraintPropagation(
          self.constraint_propagation_alpha)
    elif (self.constraint_name ==
          ConstraintName.NormalizedConstraintPropagation):
      self.constraint_operator = ConstraintPropagation(
          self.constraint_propagation_alpha, normalize=True)
    else:
      raise ValueError("Unsupported constraint name: {}".format(
          self.constraint_name))


class ConstraintOperation(metaclass=abc.ABCMeta):
  """Constraint operation class."""

  def check_input(self, affinity: np.ndarray, constraint_matrix: np.ndarray):
    """Check the input to the adjust_affinity method.

    Args:
      affinity: the input affinity matrix.
      constraint_matrix: numpy array of shape (n_samples, n_samples). The
        constraint matrix with prior information

    Raises:
      ValueError: if affinity or constraint matrix has wrong shape, etc.
    """
    if len(affinity.shape) != 2:
      raise ValueError("affinity must be 2-dimensional")
    if affinity.shape[0] != affinity.shape[1]:
      raise ValueError("affinity must be a square matrix")
    if len(constraint_matrix.shape) != 2:
      raise ValueError("constraint matrix must be 2-dimensional")
    if constraint_matrix.shape[0] != constraint_matrix.shape[1]:
      raise ValueError("constraint matrix must be a square matrix")
    if affinity.shape != constraint_matrix.shape:
      raise ValueError(
          "affinity and constraint matrix must have the same shape")

  @abc.abstractmethod
  def adjust_affinity(self,
                      affinity: np.ndarray,
                      constraint_matrix: np.ndarray):
    """An abstract method to perform the constraint operation.

    Args:
      affinity: the affinity matrix, of size (n_samples, n_samples)
      constraint_matrix: numpy array of shape (n_samples, n_samples). The
        constraint matrix with prior information

    Returns:
      a matrix of the same size as affinity
    """
    pass


class AffinityIntegration(ConstraintOperation):
  """The Affinity Integration method.

  Basic operations to integrate the affinity matrix with given pairwise
  constraints in the constraint matrix. Current integration types include
  `Max`, `Average`, and `Smooth`.
  """

  def __init__(self, integration_type: IntegrationType = IntegrationType.Max):
    self.integration_type = integration_type

  def adjust_affinity(self,
                      affinity: np.ndarray,
                      constraint_matrix: np.ndarray) -> np.ndarray:
    """Adjust the affinity matrix with constraints."""
    self.check_input(affinity, constraint_matrix)
    if self.integration_type == IntegrationType.Max:
      # For non-negative entries (Must-Link or unconstrained), take the
      # element-wise maximum. For negative entries (Cannot-Link), scale
      # affinity down by (1 + constraint_matrix) * affinity.
      return np.where(
          constraint_matrix >= 0,
          np.maximum(affinity, constraint_matrix),
          np.clip((1.0 + constraint_matrix) * affinity, 0.0, 1.0))
    elif self.integration_type == IntegrationType.Average:
      return 0.5 * (affinity + constraint_matrix)
    elif self.integration_type == IntegrationType.Smooth:
      return np.where(
          constraint_matrix >= 0,
          1.0 - (1.0 - constraint_matrix) * (1.0 - affinity),
          (1.0 + constraint_matrix) * affinity)
    else:
      raise ValueError("Unsupported integration type: {}".format(
          self.integration_type))


class ConstraintPropagation(ConstraintOperation):
  """The Constraint Propagation method.

  The pairwise constraints are firstly propagated throughout the whole graph by
  two independent horizontal and vertical propagations. The final propagated
  constraint matrix is applied to adjust the affinity matrix.

  Reference:
  [1] Lu, Zhiwu, and IP, Horace HS. "Constrained spectral clustering via
  exhaustive and efficient constraint propagation." ECCV 2010
  [2] Lu, Zhiwu, and Peng, Yuxin. "Exhaustive and efficient constraint
  propagation: A graph-based learning approach and its applications." IJCV 2013
  """

  def __init__(self, alpha: float = 0.6, normalize: bool = False):
    if alpha <= 0 or alpha >= 1:
      raise ValueError("alpha must be in (0, 1)")
    self.alpha = alpha
    self.normalize = normalize

  def adjust_affinity(self,
                      affinity: np.ndarray,
                      constraint_matrix: np.ndarray) -> np.ndarray:
    """Adjust the affinity matrix with constraints."""
    self.check_input(affinity, constraint_matrix)
    propagation_affinity = np.copy(affinity).astype(float)
    if self.normalize:
      np.fill_diagonal(propagation_affinity, 0.0)
    degree = np.sum(propagation_affinity, axis=1)
    degree_norm = 1.0 / (np.sqrt(degree) + EPS)
    # Compute affinity_norm as D^(-1/2)AD^(-1/2)
    affinity_norm = (
        degree_norm[:, None] * propagation_affinity * degree_norm[None, :])
    # The closed form of the final converged constraint matrix is:
    # (1-alpha)^2 * (I-alpha*affinity_norm)^(-1) * constraint_matrix *
    # (I-alpha*affinity_norm)^(-1). We save (I-alpha*affinity_norm)^(-1) as a
    # `temp_value` for readability
    temp_value = np.linalg.inv(
        np.eye(affinity.shape[0]) - self.alpha * affinity_norm)
    if self.normalize:
      final_constraint_matrix = (
          1 - self.alpha)**2 * temp_value.dot(
              constraint_matrix).dot(temp_value.T)
      max_abs = np.max(np.abs(final_constraint_matrix))
      if max_abs > EPS:
        final_constraint_matrix = final_constraint_matrix / max_abs
      has_init = constraint_matrix != 0
      final_constraint_matrix = np.where(
          has_init & (
              np.abs(constraint_matrix) > np.abs(final_constraint_matrix)),
          constraint_matrix,
          final_constraint_matrix)
      final_constraint_matrix = np.clip(final_constraint_matrix, -1.0, 1.0)
    else:
      final_constraint_matrix = (
          1 - self.alpha)**2 * temp_value.dot(constraint_matrix).dot(temp_value)
    # `is_positive` is a mask matrix where values of the final_constraint_matrix
    # are positive. The affinity matrix is adjusted by the final constraint
    # matrix using equation (4) in reference paper [1]
    is_positive = final_constraint_matrix > 0
    affinity1 = 1 - (1 - final_constraint_matrix * is_positive) * (
        1 - affinity * is_positive)
    affinity2 = (1 + final_constraint_matrix * np.invert(is_positive)) * (
        affinity * np.invert(is_positive))
    adjusted_affinity = affinity1 + affinity2
    if self.normalize:
      adjusted_affinity = np.clip(adjusted_affinity, 0.0, 1.0)
      np.fill_diagonal(adjusted_affinity, np.diag(affinity))
    return adjusted_affinity


class ConstraintMatrix:
  """Constraint Matrix class."""

  def __init__(self,
               speaker_turn_scores: typing.Sequence[float],
               threshold: float = 1):
    """Initialization of the constraint matrix arguments.

    Args:
      speaker_turn_scores: A list of speaker turn confidence scores. All score
        values are larger or equal to 0. If score is 0, there is no speaker
        turn. speaker_turn_scores[i+1] means the speaker turn confidence score
        between turn i+1 and turn i. The first score speaker_turn_scores[0] is
        not used.
      threshold: A threshold value for the speaker turn confidence score.
    """
    if any(score < 0 for score in speaker_turn_scores):
      raise ValueError("Speaker turn score must be larger or equal to 0.")
    self.speaker_turn_scores = speaker_turn_scores
    self.threshold = threshold

  def compute_diagonals(self) -> np.ndarray:
    """Compute diagonal constraint matrix."""
    num_turns = len(self.speaker_turn_scores)
    constraint_matrix = np.zeros((num_turns, num_turns))
    for i in range(num_turns - 1):
      speaker_turn_score = self.speaker_turn_scores[i + 1]
      if speaker_turn_score != 0:
        if speaker_turn_score > self.threshold:
          constraint_matrix[i, i + 1] = -1
          constraint_matrix[i + 1, i] = -1
      else:
        constraint_matrix[i, i + 1] = 1
        constraint_matrix[i + 1, i] = 1
    return constraint_matrix
