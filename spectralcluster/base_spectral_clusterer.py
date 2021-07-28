"""Base class to perform spectral clustering."""

import abc
from spectralcluster import refinement

RefinementName = refinement.RefinementName


class BaseSpectralClusterer:
  """A base class for spectral clustering."""

  def __init__(self,
               min_clusters=None,
               max_clusters=None,
               refinement_options=None,
               autotune=None,
               laplacian_type=None,
               stop_eigenvalue=1e-2,
               row_wise_renorm=False,
               custom_dist="cosine",
               max_iter=300):
    """Constructor of the clusterer.

    Args:
      min_clusters: minimal number of clusters allowed (only effective if not
        None)
      max_clusters: maximal number of clusters allowed (only effective if not
        None), can be used together with min_clusters to fix the number of
        clusters
      refinement_options: a RefinementOptions object that contains refinement
        arguments for the affinity matrix
      autotune: an AutoTune object to automatically search hyper-parameters
      laplacian_type: a LaplacianType. If None, we do not use a laplacian matrix
      stop_eigenvalue: when computing the number of clusters using Eigen Gap, we
        do not look at eigen values smaller than this value
      row_wise_renorm: if True, perform row-wise re-normalization on the
        spectral embeddings
      custom_dist: str or callable. custom distance measure for k-means. if a
        string, "cosine", "euclidean", "mahalanobis", or any other distance
        functions defined in scipy.spatial.distance can be used
      max_iter: the maximum number of iterations for the custom k-means
    """
    self.min_clusters = min_clusters
    self.max_clusters = max_clusters
    if not refinement_options:
      self.refinement_options = refinement.RefinementOptions()
    else:
      self.refinement_options = refinement_options
    self.autotune = autotune
    self.laplacian_type = laplacian_type
    self.row_wise_renorm = row_wise_renorm
    self.stop_eigenvalue = stop_eigenvalue
    self.custom_dist = custom_dist
    self.max_iter = max_iter

  def _get_refinement_operator(self, name):
    """Get the refinement operator for the affinity matrix.

    Args:
      name: a RefinementName

    Returns:
      object of the operator

    Raises:
      TypeError: if name is not a RefinementName
      ValueError: if name is an unknown refinement operation
    """
    if not isinstance(name, RefinementName):
      raise TypeError("name must be a RefinementName")
    elif name == RefinementName.CropDiagonal:
      return refinement.CropDiagonal()
    elif name == RefinementName.GaussianBlur:
      return refinement.GaussianBlur(
          self.refinement_options.gaussian_blur_sigma)
    elif name == RefinementName.RowWiseThreshold:
      return refinement.RowWiseThreshold(
          self.refinement_options.p_percentile,
          self.refinement_options.thresholding_soft_multiplier,
          self.refinement_options.thresholding_with_row_max,
          self.refinement_options.thresholding_with_binarization,
          self.refinement_options.thresholding_preserve_diagonal)
    elif name == RefinementName.Symmetrize:
      return refinement.Symmetrize(self.refinement_options.symmetrize_type)
    elif name == RefinementName.Diffuse:
      return refinement.Diffuse()
    elif name == RefinementName.RowWiseNormalize:
      return refinement.RowWiseNormalize()
    else:
      raise ValueError("Unknown refinement operation: {}".format(name))

  @abc.abstractmethod
  def predict(self, embeddings):
    """An abstract method for clustering speaker embeddings.

    Implementations of this method are in charge of clustering the speaker
    embeddings and assigning labels to them.

    Args:
      embeddings: numpy array of shape (n_samples, n_features)

    Returns:
      labels: numpy array of shape (n_samples,)

    Raises:
      TypeError: if embeddings has wrong type
      ValueError: if embeddings has wrong shape
    """
    pass
