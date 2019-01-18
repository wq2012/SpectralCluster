import numpy as np
import unittest
from spectralcluster import refinement


class TestCropDiagonal(unittest.TestCase):
    """Tests for the CropDiagonal class."""

    def test_3by3_matrix(self):
        X = np.array([[1, 2, 3], [3, 4, 5], [4, 2, 1]])
        Y = refinement.CropDiagonal().refine(X)
        expected = np.array([[3, 2, 3], [3, 5, 5], [4, 2, 4]])
        self.assertTrue(np.array_equal(expected, Y))


class TestDiffuse(unittest.TestCase):
    """Tests for the Diffuse class."""

    def test_2by2_matrix(self):
        X = np.array([[1, 2], [3, 4]])
        Y = refinement.Diffuse().refine(X)
        expected = np.array([[5, 11], [11, 25]])
        self.assertTrue(np.array_equal(expected, Y))


if __name__ == "__main__":
    unittest.main()
