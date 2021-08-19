import unittest
import numpy as np
from spectralcluster import autotune
from spectralcluster import configs
from spectralcluster import constraint
from spectralcluster import laplacian
from spectralcluster import refinement
from spectralcluster import spectral_clusterer
from spectralcluster import utils

RefinementName = refinement.RefinementName
ThresholdType = refinement.ThresholdType
SymmetrizeType = refinement.SymmetrizeType
LaplacianType = laplacian.LaplacianType
ConstraintName = constraint.ConstraintName
IntegrationType = constraint.IntegrationType
EigenGapType = utils.EigenGapType
ICASSP2018_REFINEMENT_SEQUENCE = configs.ICASSP2018_REFINEMENT_SEQUENCE


class TestSpectralClusterer(unittest.TestCase):
  """Tests for the SpectralClusterer class."""

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
    refinement_options = refinement.RefinementOptions(
        gaussian_blur_sigma=0,
        p_percentile=0.95,
        refinement_sequence=ICASSP2018_REFINEMENT_SEQUENCE)
    clusterer = spectral_clusterer.SpectralClusterer(
        refinement_options=refinement_options)
    labels = clusterer.predict(matrix)
    labels = utils.enforce_ordered_labels(labels)
    expected = np.array([0, 0, 1, 1, 0, 1])
    self.assertTrue(np.array_equal(expected, labels))

  def test_1000by6_matrix(self):
    matrix = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 400 +
                      [[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]] * 300 +
                      [[0.0, 0.0, 2.0, 0.0, 0.0, 0.0]] * 200 +
                      [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]] * 100)
    noisy = np.random.rand(1000, 6) * 2 - 1
    matrix = matrix + noisy * 0.1
    refinement_options = refinement.RefinementOptions(
        gaussian_blur_sigma=0,
        p_percentile=0.2,
        refinement_sequence=ICASSP2018_REFINEMENT_SEQUENCE)
    clusterer = spectral_clusterer.SpectralClusterer(
        refinement_options=refinement_options, stop_eigenvalue=0.01)
    labels = clusterer.predict(matrix)
    labels = utils.enforce_ordered_labels(labels)
    expected = np.array([0] * 400 + [1] * 300 + [2] * 200 + [3] * 100)
    self.assertTrue(np.array_equal(expected, labels))

  def test_6by2_matrix_eigengap_normalizeddiff(self):
    matrix = np.array([
        [1.0, 0.0],
        [1.1, 0.1],
        [0.0, 1.0],
        [0.1, 1.0],
        [0.9, -0.1],
        [0.0, 1.2],
    ])
    refinement_options = refinement.RefinementOptions(
        gaussian_blur_sigma=0,
        p_percentile=0.95,
        refinement_sequence=ICASSP2018_REFINEMENT_SEQUENCE)
    clusterer = spectral_clusterer.SpectralClusterer(
        refinement_options=refinement_options,
        eigengap_type=EigenGapType.NormalizedDiff)
    labels = clusterer.predict(matrix)
    labels = utils.enforce_ordered_labels(labels)
    expected = np.array([0, 0, 1, 1, 0, 1])
    self.assertTrue(np.array_equal(expected, labels))

  def test_6by2_matrix_normalized_laplacian(self):
    matrix = np.array([
        [1.0, 0.0],
        [1.1, 0.1],
        [0.0, 1.0],
        [0.1, 1.0],
        [0.9, -0.1],
        [0.0, 1.2],
    ])

    refinement_sequence = []
    refinement_options = refinement.RefinementOptions(
        p_percentile=0.95, refinement_sequence=refinement_sequence)
    clusterer = spectral_clusterer.SpectralClusterer(
        max_clusters=2,
        refinement_options=refinement_options,
        laplacian_type=LaplacianType.GraphCut,
        row_wise_renorm=True)
    labels = clusterer.predict(matrix)
    labels = utils.enforce_ordered_labels(labels)
    expected = np.array([0, 0, 1, 1, 0, 1])
    self.assertTrue(np.array_equal(expected, labels))

  def test_1000by6_matrix_normalized_laplacian(self):
    matrix = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 400 +
                      [[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]] * 300 +
                      [[0.0, 0.0, 2.0, 0.0, 0.0, 0.0]] * 200 +
                      [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]] * 100)
    noisy = np.random.rand(1000, 6) * 2 - 1
    matrix = matrix + noisy * 0.1

    refinement_sequence = []
    refinement_options = refinement.RefinementOptions(
        p_percentile=0.95, refinement_sequence=refinement_sequence)
    clusterer = spectral_clusterer.SpectralClusterer(
        max_clusters=4,
        refinement_options=refinement_options,
        laplacian_type=LaplacianType.GraphCut,
        row_wise_renorm=True)
    labels = clusterer.predict(matrix)
    labels = utils.enforce_ordered_labels(labels)
    expected = np.array([0] * 400 + [1] * 300 + [2] * 200 + [3] * 100)
    self.assertTrue(np.array_equal(expected, labels))

  def test_6by2_matrix_auto_tune(self):
    matrix = np.array([
        [1.0, 0.0],
        [1.1, 0.1],
        [0.0, 1.0],
        [0.1, 1.0],
        [0.9, -0.1],
        [0.0, 1.2],
    ])

    refinement_sequence = [RefinementName.RowWiseThreshold]
    refinement_options = refinement.RefinementOptions(
        thresholding_type=ThresholdType.Percentile,
        refinement_sequence=refinement_sequence)
    auto_tune = autotune.AutoTune(
        p_percentile_min=0.60,
        p_percentile_max=0.95,
        init_search_step=0.05,
        search_level=1)
    clusterer = spectral_clusterer.SpectralClusterer(
        max_clusters=2,
        refinement_options=refinement_options,
        autotune=auto_tune,
        laplacian_type=LaplacianType.GraphCut,
        row_wise_renorm=True)
    labels = clusterer.predict(matrix)
    labels = utils.enforce_ordered_labels(labels)
    expected = np.array([0, 0, 1, 1, 0, 1])
    self.assertTrue(np.array_equal(expected, labels))

  def test_1000by6_matrix_auto_tune(self):
    matrix = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 400 +
                      [[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]] * 300 +
                      [[0.0, 0.0, 2.0, 0.0, 0.0, 0.0]] * 200 +
                      [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]] * 100)
    noisy = np.random.rand(1000, 6) * 2 - 1
    matrix = matrix + noisy * 0.1

    refinement_sequence = [RefinementName.RowWiseThreshold]
    refinement_options = refinement.RefinementOptions(
        thresholding_type=ThresholdType.Percentile,
        refinement_sequence=refinement_sequence)
    auto_tune = autotune.AutoTune(
        p_percentile_min=0.9,
        p_percentile_max=0.95,
        init_search_step=0.03,
        search_level=1)
    clusterer = spectral_clusterer.SpectralClusterer(
        max_clusters=4,
        refinement_options=refinement_options,
        autotune=auto_tune,
        laplacian_type=LaplacianType.GraphCut,
        row_wise_renorm=True)
    labels = clusterer.predict(matrix)
    labels = utils.enforce_ordered_labels(labels)
    expected = np.array([0] * 400 + [1] * 300 + [2] * 200 + [3] * 100)
    self.assertTrue(np.array_equal(expected, labels))

  def test_6by2_matrix_affinity_integration(self):
    matrix = np.array([
        [1.0, 0.0],
        [1.1, 0.1],
        [0.0, 1.0],
        [0.1, 1.0],
        [0.9, -0.1],
        [0.0, 1.2],
    ])

    constraint_matrix = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 1],
        [0, 0, 1, 1, 1, 1],
        [0, 0, 1, 1, 1, 1],
        [0, 0, 1, 1, 1, 1],
    ])

    refinement_sequence = [
        RefinementName.RowWiseThreshold, RefinementName.Symmetrize
    ]
    refinement_options = refinement.RefinementOptions(
        p_percentile=0.95,
        thresholding_type=ThresholdType.Percentile,
        thresholding_with_binarization=True,
        thresholding_preserve_diagonal=True,
        symmetrize_type=SymmetrizeType.Average,
        refinement_sequence=refinement_sequence)
    constraint_options = constraint.ConstraintOptions(
        constraint_name=ConstraintName.AffinityIntegration,
        apply_before_refinement=False,
        integration_type=IntegrationType.Max)
    clusterer = spectral_clusterer.SpectralClusterer(
        max_clusters=2,
        refinement_options=refinement_options,
        constraint_options=constraint_options,
        laplacian_type=LaplacianType.GraphCut,
        row_wise_renorm=True)
    labels = clusterer.predict(matrix, constraint_matrix)
    labels = utils.enforce_ordered_labels(labels)
    expected = np.array([0, 0, 1, 1, 1, 1])
    self.assertTrue(np.array_equal(expected, labels))

  def test_6by2_matrix_constraint_propagation(self):
    matrix = np.array([
        [1.0, 0.0],
        [1.1, 0.1],
        [0.0, 1.0],
        [0.1, 1.0],
        [0.9, -0.1],
        [0.0, 1.2],
    ])

    constraint_matrix = np.array([
        [1, 1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, -1],
        [0, 0, 0, 0, -1, 1],
    ])
    refinement_sequence = [
        RefinementName.RowWiseThreshold, RefinementName.Symmetrize
    ]
    refinement_options = refinement.RefinementOptions(
        p_percentile=0.95,
        thresholding_type=ThresholdType.Percentile,
        thresholding_with_binarization=True,
        thresholding_preserve_diagonal=True,
        symmetrize_type=SymmetrizeType.Average,
        refinement_sequence=refinement_sequence)
    constraint_options = constraint.ConstraintOptions(
        constraint_name=ConstraintName.ConstraintPropagation,
        apply_before_refinement=True,
        constraint_propagation_alpha=0.6)
    clusterer = spectral_clusterer.SpectralClusterer(
        max_clusters=2,
        refinement_options=refinement_options,
        constraint_options=constraint_options,
        laplacian_type=LaplacianType.GraphCut,
        row_wise_renorm=True)
    labels = clusterer.predict(matrix, constraint_matrix)
    labels = utils.enforce_ordered_labels(labels)
    expected = np.array([0, 0, 1, 1, 0, 1])
    self.assertTrue(np.array_equal(expected, labels))


if __name__ == "__main__":
  unittest.main()
