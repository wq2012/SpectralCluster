from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import unittest
from spectralcluster import spectral_clusterer


class TestSpectralClusterer(unittest.TestCase):
    """Tests for the SpectralClusterer class."""

    def setUp(self):
        # fix random seeds for reproducing results
        np.random.seed(1)

    def test_6by2_matrix(self):
        X = np.array([
            [1.0, 0.0],
            [1.1, 0.1],
            [0.0, 1.0],
            [0.1, 1.0],
            [0.9, -0.1],
            [0.0, 1.2],
        ])
        clusterer = spectral_clusterer.SpectralClusterer(
            p_percentile=0.95,
            gaussian_blur_sigma=0)
        labels = clusterer.predict(X)
        expected = np.array([0, 0, 1, 1, 0, 1])
        self.assertTrue(np.array_equal(expected, labels))

    def test_1000by6_matrix(self):
        X = np.array(
            [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 400 +
            [[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]] * 300 +
            [[0.0, 0.0, 2.0, 0.0, 0.0, 0.0]] * 200 +
            [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]] * 100
        )
        noisy = np.random.rand(1000, 6) * 2 - 1
        X = X + noisy * 0.1
        clusterer = spectral_clusterer.SpectralClusterer(
            p_percentile=0.2,
            gaussian_blur_sigma=0,
            stop_eigenvalue=0.01)
        labels = clusterer.predict(X)
        expected = np.array(
            [2] * 400 + [0] * 300 + [1] * 200 + [3] * 100
        )
        self.assertTrue(np.array_equal(expected, labels))


if __name__ == "__main__":
    unittest.main()
