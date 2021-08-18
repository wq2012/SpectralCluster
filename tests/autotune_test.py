import unittest
import numpy as np
from spectralcluster import autotune
from spectralcluster import laplacian
from spectralcluster import refinement
from spectralcluster import spectral_clusterer
from spectralcluster import utils

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
    self.assertTrue(
        np.allclose(
            np.array(p_percentile_range), np.array(expected), atol=0.01))

  def test_update_percentile_range(self):
    auto_tune = autotune.AutoTune(
        p_percentile_min=0.4,
        p_percentile_max=0.9,
        init_search_step=0.1,
        search_level=1)
    p_percentile_range = auto_tune.update_percentile_range(0.5, 0.8, 0.05)
    expected = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    self.assertTrue(
        np.allclose(
            np.array(p_percentile_range), np.array(expected), atol=0.01))

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

    def p_percentile_to_ratio(p_percentile):
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


if __name__ == "__main__":
  unittest.main()
