import numpy as np
import unittest
from spectralcluster import refinement


class TestDiffuse(unittest.TestCase):
    """Tests for the Diffuse class."""

    def test_2by2_matrix(self):
        X = np.array([[1, 2], [3, 4]])
        op = refinement.Diffuse()
        Y = op(X)
        expected = np.array([[5, 11], [11, 21]])
        self.assertTrue(np.array_equal(expected, Y))
