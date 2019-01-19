from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import unittest
from spectralcluster import spectral_clusterer


class TestSpectralClusterer(unittest.TestCase):
    """Tests for the SpectralClusterer class."""

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
        labels = clusterer.cluster(X)
        expected = np.array([0, 0, 1, 1, 0, 1])
        self.assertTrue(np.array_equal(expected, labels))


if __name__ == "__main__":
    unittest.main()
