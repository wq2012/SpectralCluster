import unittest
import numpy as np
from spectralcluster import autotune
from spectralcluster import laplacian
from spectralcluster import refinement
from spectralcluster import spectral_clusterer
from spectralcluster import utils
import typing

RefinementName = refinement.RefinementName
ThresholdType = refinement.ThresholdType
LaplacianType = laplacian.LaplacianType


class TestAutotune(unittest.TestCase):
  """Tests for the AutoTune class."""

  def test_get_percentile_range(self):
    auto_tune = autotune.AutoTune(
        p_percentile_min=0.60,
        p_percentile_max=0.66,
        init_search_step=0.01,
        search_level=1)
    p_percentile_range = auto_tune.get_percentile_range()
    expected = [0.60, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66]
    np.testing.assert_allclose(
            np.array(p_percentile_range), np.array(expected), atol=0.01)

  def test_update_percentile_range(self):
    auto_tune = autotune.AutoTune(
        p_percentile_min=0.4,
        p_percentile_max=0.9,
        init_search_step=0.1,
        search_level=1)
    p_percentile_range = auto_tune.update_percentile_range(0.5, 0.8, 0.05)
    expected = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    np.testing.assert_allclose(
            np.array(p_percentile_range), np.array(expected), atol=0.01)

  def test_6by2matrix_tune(self):
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

    affinity = utils.compute_affinity_matrix(matrix)

    def p_percentile_to_ratio(p_percentile: float) -> (
        typing.Tuple[float, np.ndarray, int]):
      """compute the `ratio` given a `p_percentile` value."""
      clusterer.refinement_options.p_percentile = p_percentile
      (eigenvectors, n_clusters,
       max_delta_norm) = clusterer._compute_eigenvectors_ncluster(affinity)
      ratio = (1 - p_percentile) / max_delta_norm
      return ratio, eigenvectors, n_clusters

    eigenvectors, n_clusters, p_percentile = clusterer.autotune.tune(
        p_percentile_to_ratio)

    self.assertEqual((6, 6), eigenvectors.shape)
    self.assertEqual(n_clusters, 2)
    self.assertEqual(p_percentile, 0.6)

    # Calling tune() a second time should produce the exact same result.
    eigenvectors2, n_clusters2, p_percentile2 = clusterer.autotune.tune(
        p_percentile_to_ratio)
    np.testing.assert_equal(eigenvectors, eigenvectors2)
    self.assertEqual(n_clusters, n_clusters2)
    self.assertEqual(p_percentile, p_percentile2)

  def test_multi_level_search_preserves_optimum(self):
    auto_tune = autotune.AutoTune(
        p_percentile_min=0.50,
        p_percentile_max=0.90,
        init_search_step=0.10,
        search_level=3)
    call_count = 0

    def mock_p_percentile_to_ratio(p_percentile: float):
      nonlocal call_count
      call_count += 1
      # Minimum ratio is at p_percentile = 0.70 (already in level-1 grid).
      ratio = abs(p_percentile - 0.70) + 0.1
      return ratio, np.eye(4), 2

    _, n_clusters, best_p = auto_tune.tune(mock_p_percentile_to_ratio)
    self.assertEqual(n_clusters, 2)
    self.assertAlmostEqual(best_p, 0.70, places=5)

  def test_invalid_proxy_and_early_stop(self):
    with self.assertRaises(TypeError):
      autotune.AutoTune(proxy="invalid")

    auto_tune = autotune.AutoTune(
        p_percentile_min=0.60,
        p_percentile_max=0.6001,
        init_search_step=1e-5,
        search_level=3)
    _, n_clusters, best_p = auto_tune.tune(
        lambda p: (1.0, np.eye(2), 2))
    self.assertEqual(n_clusters, 2)
    self.assertAlmostEqual(best_p, 0.60, places=3)


if __name__ == "__main__":
  unittest.main()
