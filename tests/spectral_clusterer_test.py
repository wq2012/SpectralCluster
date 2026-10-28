import unittest
import numpy as np
from spectralcluster import autotune
from spectralcluster import configs
from spectralcluster import constraint
from spectralcluster import fallback_clusterer
from spectralcluster import laplacian
from spectralcluster import refinement
from spectralcluster import spectral_clusterer
from spectralcluster import utils

AutoTuneProxy = autotune.AutoTuneProxy
RefinementName = refinement.RefinementName
ThresholdType = refinement.ThresholdType
SymmetrizeType = refinement.SymmetrizeType
LaplacianType = laplacian.LaplacianType
ConstraintName = constraint.ConstraintName
IntegrationType = constraint.IntegrationType
EigenGapType = utils.EigenGapType
FallbackOptions = fallback_clusterer.FallbackOptions
SingleClusterCondition = fallback_clusterer.SingleClusterCondition
FallbackClustererType = fallback_clusterer.FallbackClustererType
ICASSP2018_REFINEMENT_SEQUENCE = configs.ICASSP2018_REFINEMENT_SEQUENCE


class TestSpectralClusterer(unittest.TestCase):
  """Tests for the SpectralClusterer class."""

  def setUp(self):
    super().setUp()
    np.random.seed(0)

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
    np.testing.assert_equal(expected, labels)

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
    np.testing.assert_equal(expected, labels)

  def test_1000by6_matrix_reduce_dimension(self):
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
        refinement_options=refinement_options,
        stop_eigenvalue=0.01,
        max_spectral_size=100)
    labels = clusterer.predict(matrix)
    labels = utils.enforce_ordered_labels(labels)
    expected = np.array([0] * 400 + [1] * 300 + [2] * 200 + [3] * 100)
    np.testing.assert_equal(expected, labels)

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
    np.testing.assert_equal(expected, labels)

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
    np.testing.assert_equal(expected, labels)

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
    np.testing.assert_equal(expected, labels)

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
    np.testing.assert_equal(expected, labels)

  def test_2by2_matrix_auto_tune(self):
    matrix = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    refinement_sequence = [RefinementName.RowWiseThreshold]
    refinement_options = refinement.RefinementOptions(
        thresholding_type=ThresholdType.Percentile,
        refinement_sequence=refinement_sequence)
    auto_tune = autotune.AutoTune(
        p_percentile_min=0.60,
        p_percentile_max=0.95,
        init_search_step=0.05,
        search_level=1,
        proxy=AutoTuneProxy.PercentileOverNME)
    fallback_options = fallback_clusterer.FallbackOptions(
        spectral_min_embeddings=3)
    clusterer = spectral_clusterer.SpectralClusterer(
        max_clusters=2,
        refinement_options=refinement_options,
        autotune=auto_tune,
        fallback_options=fallback_options,
        laplacian_type=LaplacianType.GraphCut,
        row_wise_renorm=True)
    labels = clusterer.predict(matrix)
    labels = utils.enforce_ordered_labels(labels)
    expected = np.array([0, 1])
    np.testing.assert_equal(expected, labels)

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
    np.testing.assert_equal(expected, labels)

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
    np.testing.assert_equal(expected, labels)

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
    np.testing.assert_equal(expected, labels)

  def test_6by2_matrix_single_cluster(self):
    matrix = np.array([
        [1.0, 0.0],
        [1.1, 0.1],
        [1.0, 0.0],
        [1.1, 0.0],
        [0.9, -0.1],
        [1.0, 0.2],
    ])
    refinement_options = refinement.RefinementOptions(
        gaussian_blur_sigma=0,
        p_percentile=0.95,
        refinement_sequence=ICASSP2018_REFINEMENT_SEQUENCE)
    clusterer = spectral_clusterer.SpectralClusterer(
        min_clusters=1,
        refinement_options=refinement_options)
    labels = clusterer.predict(matrix)
    labels = utils.enforce_ordered_labels(labels)
    expected = np.array([0, 0, 0, 0, 0, 0])
    np.testing.assert_equal(expected, labels)

  def test_6by2_matrix_single_cluster_all_affinity(self):
    matrix = np.array([
        [1.0, 0.0],
        [1.1, 0.1],
        [1.0, 0.0],
        [1.1, 0.0],
        [0.9, -0.1],
        [1.0, 0.5],
    ])
    # High threshold.
    fallback_options = FallbackOptions(
        single_cluster_condition=SingleClusterCondition.AllAffinity,
        single_cluster_affinity_threshold=0.93)
    clusterer = spectral_clusterer.SpectralClusterer(
        min_clusters=1,
        laplacian_type=LaplacianType.GraphCut,
        refinement_options=None,
        fallback_options=fallback_options)
    labels = clusterer.predict(matrix)
    labels = utils.enforce_ordered_labels(labels)
    expected = np.array([0, 0, 0, 0, 0, 1])
    np.testing.assert_equal(expected, labels)

    # Low threshold.
    fallback_options = FallbackOptions(
        single_cluster_condition=SingleClusterCondition.AllAffinity,
        single_cluster_affinity_threshold=0.91)
    clusterer = spectral_clusterer.SpectralClusterer(
        min_clusters=1,
        laplacian_type=LaplacianType.GraphCut,
        refinement_options=None,
        fallback_options=fallback_options)
    labels = clusterer.predict(matrix)
    labels = utils.enforce_ordered_labels(labels)
    expected = np.array([0, 0, 0, 0, 0, 0])
    np.testing.assert_equal(expected, labels)

  def test_6by2_matrix_single_cluster_neighbor_affinity(self):
    matrix = np.array([
        [1.0, 0.0],
        [1.1, 0.1],
        [1.0, 0.0],
        [1.0, 0.5],
        [1.1, 0.0],
        [0.9, -0.1],
    ])
    # High threshold.
    fallback_options = FallbackOptions(
        single_cluster_condition=SingleClusterCondition.NeighborAffinity,
        single_cluster_affinity_threshold=0.96)
    clusterer = spectral_clusterer.SpectralClusterer(
        min_clusters=1,
        laplacian_type=LaplacianType.GraphCut,
        refinement_options=None,
        fallback_options=fallback_options)
    labels = clusterer.predict(matrix)
    labels = utils.enforce_ordered_labels(labels)
    expected = np.array([0, 0, 0, 1, 0, 0])
    np.testing.assert_equal(expected, labels)

    # Low threshold.
    fallback_options = FallbackOptions(
        single_cluster_condition=SingleClusterCondition.NeighborAffinity,
        single_cluster_affinity_threshold=0.94)
    clusterer = spectral_clusterer.SpectralClusterer(
        min_clusters=1,
        laplacian_type=LaplacianType.GraphCut,
        refinement_options=None,
        fallback_options=fallback_options)
    labels = clusterer.predict(matrix)
    labels = utils.enforce_ordered_labels(labels)
    expected = np.array([0, 0, 0, 0, 0, 0])
    np.testing.assert_equal(expected, labels)

  def test_6by2_matrix_single_cluster_affinity_std(self):
    matrix = np.array([
        [1.0, 0.0],
        [1.1, 0.1],
        [1.0, 0.0],
        [1.0, 0.5],
        [1.1, 0.0],
        [0.9, -0.1],
    ])
    # Low threshold.
    fallback_options = FallbackOptions(
        single_cluster_condition=SingleClusterCondition.AffinityStd,
        single_cluster_affinity_threshold=0.02)
    clusterer = spectral_clusterer.SpectralClusterer(
        min_clusters=1,
        laplacian_type=LaplacianType.GraphCut,
        refinement_options=None,
        fallback_options=fallback_options)
    labels = clusterer.predict(matrix)
    labels = utils.enforce_ordered_labels(labels)
    expected = np.array([0, 0, 0, 1, 0, 0])
    np.testing.assert_equal(expected, labels)

    # High threshold.
    fallback_options = FallbackOptions(
        single_cluster_condition=SingleClusterCondition.AffinityStd,
        single_cluster_affinity_threshold=0.03)
    clusterer = spectral_clusterer.SpectralClusterer(
        min_clusters=1,
        laplacian_type=LaplacianType.GraphCut,
        refinement_options=None,
        fallback_options=fallback_options)
    labels = clusterer.predict(matrix)
    labels = utils.enforce_ordered_labels(labels)
    expected = np.array([0, 0, 0, 0, 0, 0])
    np.testing.assert_equal(expected, labels)

  def test_6by2_matrix_single_cluster_fallback_naive(self):
    matrix = np.array([
        [1.0, 0.0],
        [1.1, 0.1],
        [1.0, 0.0],
        [1.0, 0.5],
        [1.1, 0.0],
        [0.9, -0.1],
    ])
    # High threshold.
    fallback_options = FallbackOptions(
        single_cluster_condition=SingleClusterCondition.FallbackClusterer,
        fallback_clusterer_type=FallbackClustererType.Naive,
        naive_threshold=0.95)
    clusterer = spectral_clusterer.SpectralClusterer(
        min_clusters=1,
        laplacian_type=LaplacianType.GraphCut,
        refinement_options=None,
        fallback_options=fallback_options)
    labels = clusterer.predict(matrix)
    labels = utils.enforce_ordered_labels(labels)
    expected = np.array([0, 0, 0, 1, 0, 0])
    np.testing.assert_equal(expected, labels)

    # Low threshold.
    fallback_options = FallbackOptions(
        single_cluster_condition=SingleClusterCondition.FallbackClusterer,
        fallback_clusterer_type=FallbackClustererType.Naive,
        naive_threshold=0.9)
    clusterer = spectral_clusterer.SpectralClusterer(
        min_clusters=1,
        laplacian_type=LaplacianType.GraphCut,
        refinement_options=None,
        fallback_options=fallback_options)
    labels = clusterer.predict(matrix)
    labels = utils.enforce_ordered_labels(labels)
    expected = np.array([0, 0, 0, 0, 0, 0])
    np.testing.assert_equal(expected, labels)

  def test_fit_predict_and_predict_with_affinity(self):
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
        refinement_options=refinement_options, random_state=42)
    labels_fit = utils.enforce_ordered_labels(clusterer.fit_predict(matrix))
    affinity = utils.compute_affinity_matrix(matrix)
    labels_aff = utils.enforce_ordered_labels(
        clusterer.predict_with_affinity(affinity))
    expected = np.array([0, 0, 1, 1, 0, 1])
    np.testing.assert_equal(expected, labels_fit)
    np.testing.assert_equal(expected, labels_aff)

    with self.assertRaises(TypeError):
      clusterer.predict_with_affinity([[1.0, 0.0], [0.0, 1.0]])
    with self.assertRaises(ValueError):
      clusterer.predict_with_affinity(np.zeros((2, 3)))

    fallback_options = FallbackOptions(
        single_cluster_condition=SingleClusterCondition.FallbackClusterer)
    single_clusterer = spectral_clusterer.SpectralClusterer(
        min_clusters=1, fallback_options=fallback_options)
    with self.assertRaises(ValueError):
      single_clusterer.predict_with_affinity(affinity)

  def test_percentile_over_nme_and_validation_errors(self):
    matrix = np.array([
        [1.0, 0.0],
        [1.1, 0.1],
        [0.0, 1.0],
        [0.1, 1.0],
        [0.9, -0.1],
        [0.0, 1.2],
    ])
    refinement_options = refinement.RefinementOptions(
        thresholding_type=ThresholdType.Percentile,
        refinement_sequence=[RefinementName.RowWiseThreshold])
    auto_tune = autotune.AutoTune(
        p_percentile_min=0.60,
        p_percentile_max=0.95,
        init_search_step=0.05,
        search_level=1,
        proxy=AutoTuneProxy.PercentileOverNME)
    clusterer = spectral_clusterer.SpectralClusterer(
        max_clusters=2,
        refinement_options=refinement_options,
        autotune=auto_tune,
        laplacian_type=LaplacianType.GraphCut,
        row_wise_renorm=True)
    labels = utils.enforce_ordered_labels(clusterer.predict(matrix))
    np.testing.assert_equal(np.array([0, 0, 1, 1, 0, 1]), labels)

    with self.assertRaises(TypeError):
      clusterer.predict([[1.0, 0.0]])
    with self.assertRaises(ValueError):
      clusterer.predict(np.array([1.0, 0.0]))

    bad_size_clusterer = spectral_clusterer.SpectralClusterer(
        max_clusters=4, max_spectral_size=3)
    with self.assertRaises(ValueError):
      bad_size_clusterer.predict(matrix)
    with self.assertRaises(RuntimeError):
      bad_size_clusterer.predict(matrix, constraint_matrix=np.eye(6))

    no_thresh_clusterer = spectral_clusterer.SpectralClusterer(
        autotune=auto_tune,
        refinement_options=refinement.RefinementOptions(refinement_sequence=[]))
    with self.assertRaises(ValueError):
      no_thresh_clusterer.predict(matrix)

    custom_clusterer = spectral_clusterer.SpectralClusterer(
        max_clusters=2,
        post_eigen_cluster_function=lambda **kwargs: np.zeros(
            kwargs["spectral_embeddings"].shape[0], dtype=int))
    np.testing.assert_equal(np.zeros(6, dtype=int),
                            custom_clusterer.predict(matrix))


if __name__ == "__main__":
  unittest.main()
