import unittest

import numpy as np
from spectralcluster import configs
from spectralcluster import multi_stage_clusterer
from spectralcluster import refinement
from spectralcluster import spectral_clusterer
from spectralcluster import utils


class TestMultiStageClusterer(unittest.TestCase):
  """Tests for the MultiStageClusterer class."""

  def setUp(self):
    super().setUp()
    refinement_options = refinement.RefinementOptions(
        gaussian_blur_sigma=0,
        p_percentile=0.95,
        refinement_sequence=configs.ICASSP2018_REFINEMENT_SEQUENCE)
    main_clusterer = spectral_clusterer.SpectralClusterer(
        refinement_options=refinement_options)
    self.multi_stage = multi_stage_clusterer.MultiStageClusterer(
      main_clusterer=main_clusterer,
      fallback_threshold=0.5,
      L=3,
      U1=5,
      U2=7
    )

  def test_single_input(self):
    embedding = np.array([[1, 2]])
    labels = self.multi_stage.streaming_predict(embedding)
    expected = np.array([0])
    np.testing.assert_equal(expected, labels)

  def test_fallback(self):
    embeddings = [[1, 2], [3, -1]]
    for embedding in embeddings:
      labels = self.multi_stage.streaming_predict(np.array(embedding))
    labels = utils.enforce_ordered_labels(labels)
    expected = np.array([0, 1])
    np.testing.assert_equal(expected, labels)

  def test_main(self):
    embeddings = [
      [1, 2],
      [3, -1],
      [1, 1],
      [-2, -1],
      ]
    for embedding in embeddings:
      labels = self.multi_stage.streaming_predict(np.array(embedding))
    labels = utils.enforce_ordered_labels(labels)
    expected = np.array([0, 0, 0, 1])
    np.testing.assert_equal(expected, labels)

  def test_pre(self):
    embeddings = [
      [1, 2],
      [3, -1],
      [1, 1],
      [-2, -1],
      [0, 1],
      [-2, 0],
      ]
    for embedding in embeddings:
      labels = self.multi_stage.streaming_predict(np.array(embedding))
    labels = utils.enforce_ordered_labels(labels)
    expected = np.array([0, 1, 0, 2, 3, 2])
    np.testing.assert_equal(expected, labels)

  def test_compression(self):
    embeddings = [
      [1, 2],
      [3, -1],
      [1, 1],
      [-2, -1],
      [0, 1],
      [-2, 0],
      [1, 2],
      [3, -1],
      ]
    for embedding in embeddings:
      labels = self.multi_stage.streaming_predict(np.array(embedding))
    labels = utils.enforce_ordered_labels(labels)
    expected = np.array([0, 1, 0, 2, 3, 2, 0, 1])
    np.testing.assert_equal(expected, labels)

  def test_double_compression(self):
    embeddings = [
      [1, 2],
      [3, -1],
      [1, 1],
      [-2, -1],
      [0, 1],
      [-2, 0],
      [1, 2],
      [3, -1],
      [1, 1],
      [-2, -1],
      ]
    for embedding in embeddings:
      labels = self.multi_stage.streaming_predict(np.array(embedding))
    labels = utils.enforce_ordered_labels(labels)
    expected = np.array([0, 1, 0, 2, 3, 2, 0, 1, 0, 2])
    np.testing.assert_equal(expected, labels)

  def test_many_compression(self):
    embeddings = [
      [1, 2],
      [3, -1],
      [1, 1],
      [-2, -1],
      [0, 1],
      [-2, 0],
      [1, 2],
      [3, -1],
      [1, 1],
      [-2, -1],
      [0, 1],
      [-2, 0],
      [1, 2],
      [3, -1],
      [1, 1],
      [-2, -1],
      ]
    for embedding in embeddings:
      labels = self.multi_stage.streaming_predict(np.array(embedding))
    labels = utils.enforce_ordered_labels(labels)
    expected = np.array([0, 1, 0, 2, 3, 2, 0, 1, 0, 2, 3, 2, 0, 1, 0, 2])
    np.testing.assert_equal(expected, labels)

  def test_1000by6_matrix(self):
    matrix = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 100 +
                      [[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]] * 200 +
                      [[0.0, 0.0, 2.0, 0.0, 0.0, 0.0]] * 300 +
                      [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]] * 400)
    noisy = np.random.rand(1000, 6) * 2 - 1
    matrix = matrix + noisy * 0.1
    refinement_options = refinement.RefinementOptions(
        gaussian_blur_sigma=0,
        p_percentile=0.2,
        refinement_sequence=configs.ICASSP2018_REFINEMENT_SEQUENCE)
    main_clusterer = spectral_clusterer.SpectralClusterer(
        refinement_options=refinement_options, stop_eigenvalue=0.01)
    multi_stage = multi_stage_clusterer.MultiStageClusterer(
      main_clusterer=main_clusterer,
      fallback_threshold=0.5,
      L=50,
      U1=200,
      U2=400
    )
    for embedding in matrix:
      labels = multi_stage.streaming_predict(embedding)
    labels = utils.enforce_ordered_labels(labels)
    expected = np.array([0] * 100 + [1] * 200 + [2] * 300 + [3] * 400)
    np.testing.assert_equal(expected, labels)


if __name__ == "__main__":
  unittest.main()