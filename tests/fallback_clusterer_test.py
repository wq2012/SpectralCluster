import unittest
import numpy as np
from spectralcluster import fallback_clusterer
from spectralcluster import utils

FallbackOptions = fallback_clusterer.FallbackOptions
FallbackClusterer = fallback_clusterer.FallbackClusterer
SingleClusterCondition = fallback_clusterer.SingleClusterCondition


class TestFallbackClusterer(unittest.TestCase):

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
    options = FallbackOptions()
    clusterer = FallbackClusterer(options)
    labels = clusterer.predict(matrix)
    labels = utils.enforce_ordered_labels(labels)
    expected = np.array([0, 0, 1, 1, 0, 1])
    np.testing.assert_equal(expected, labels)


class TestCheckSingleCluster(unittest.TestCase):

  def test_affinity_gmm_bic1(self):
    affinity = np.array([[1, 0.999, 1.001], [0.999, 1, 1], [1.001, 1, 1]])
    fallback_options = fallback_clusterer.FallbackOptions(
        single_cluster_condition=SingleClusterCondition.AffinityGmmBic)
    self.assertTrue(
        fallback_clusterer.check_single_cluster(
            fallback_options, None, affinity))

  def test_affinity_gmm_bic2(self):
    affinity = np.array([[1, 2, 2], [2, 1, 1], [2, 1, 1]])
    fallback_options = fallback_clusterer.FallbackOptions(
        single_cluster_condition=SingleClusterCondition.AffinityGmmBic)
    self.assertFalse(
        fallback_clusterer.check_single_cluster(
            fallback_options, None, affinity))


if __name__ == "__main__":
  unittest.main()
