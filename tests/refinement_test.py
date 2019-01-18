import numpy as np
import unittest
from spectralcluster import refinement


class TestDiffuse(unittest.TestCase):
    """Tests for the Diffuse class."""

    def test_2by2_matrix(self):
        X = np.array([[1, 2], [3, 4]])
        Y = refinement.Diffuse().refine(X)
        expected = np.array([[5, 11], [11, 25]])
        self.assertTrue(np.array_equal(expected, Y))

if __name__ == "__main__":
    unittest.main()
