import unittest
import numpy as np
from spectralcluster import autotune
from spectralcluster import refinement
from spectralcluster import spectral_clusterer
from spectralcluster import utils


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
        gaussian_blur_sigma=0, p_percentile=0.95)
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
        gaussian_blur_sigma=0, p_percentile=0.2)
    clusterer = spectral_clusterer.SpectralClusterer(
        refinement_options=refinement_options, stop_eigenvalue=0.01)
    labels = clusterer.predict(matrix)
    labels = utils.enforce_ordered_labels(labels)
    expected = np.array([0] * 400 + [1] * 300 + [2] * 200 + [3] * 100)
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
        laplacian_type="graph_cut",
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
        laplacian_type="graph_cut",
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

    refinement_sequence = []
    refinement_options = refinement.RefinementOptions(
        use_autotune=True,
        thresholding_with_row_max=False,
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
        laplacian_type="graph_cut",
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

    refinement_sequence = []
    refinement_options = refinement.RefinementOptions(
        use_autotune=True,
        thresholding_with_row_max=False,
        refinement_sequence=refinement_sequence)
    auto_tune = autotune.AutoTune(
        p_percentile_min=0.85,
        p_percentile_max=0.95,
        init_search_step=0.05,
        search_level=1)
    clusterer = spectral_clusterer.SpectralClusterer(
        max_clusters=4,
        refinement_options=refinement_options,
        autotune=auto_tune,
        laplacian_type="graph_cut",
        row_wise_renorm=True)
    labels = clusterer.predict(matrix)
    labels = utils.enforce_ordered_labels(labels)
    expected = np.array([0] * 400 + [1] * 300 + [2] * 200 + [3] * 100)
    self.assertTrue(np.array_equal(expected, labels))


if __name__ == "__main__":
  unittest.main()
