"""In some cases, we use a fallback clusterer instead of spectral.

Spectral clustering exploits the global structure of the data. But there are
cases where spectral clustering does not work as well as some other simpler
clustering methods, such as when the number of embeddings is too small.
"""

import enum
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture


class SingleClusterCondition(enum.Enum):
  """Which condition do we use for deciding single-vs-multi cluster(s)."""

  # Fit affinity values with GMM with 1-vs-2 component(s), and use
  # Bayesian Information Criterion (BIC) to decide whether there are
  # at least two clusters.
  # Note that this approach does not require additional parameters.
  AffinityGmmBic = enum.auto()

  # If all affinities are larger than threshold, there is only a single cluster.
  AllAffinity = enum.auto()

  # If all neighboring affinities are larger than threshold, there is only
  # a single cluster.
  NeighborAffinity = enum.auto()

  # If the standard deviation of all affinities is smaller than threshold,
  # there is only a single cluster.
  AffinityStd = enum.auto()

  # Use fallback clusterer to make the decision. If fallback clusterer
  # finds multiple clusters, continue with spectral clusterer.
  # TODO: currently AgglomerativeClustering is hardcoded to 2 clusters, so this
  # is not useful. We will add other fallback clusterers in the future.
  FallbackClusterer = enum.auto()


class FallbackOptions:
  """Options for fallback options."""

  def __init__(self,
               spectral_min_embeddings=1,
               single_cluster_condition=SingleClusterCondition.AffinityGmmBic,
               single_cluster_affinity_threshold=1.0):
    """Initialization of the fallback options.

    Args:
      spectral_min_embeddings: we only run spectral clusterer if we have at
        least these many embeddings; otherwise we run fallback clusterer
      single_cluster_condition: how do we decide single-vs-multi cluster(s)
      single_cluster_affinity_threshold: affinity threshold to decide
        whether there is only a single cluster
    """
    self.spectral_min_embeddings = spectral_min_embeddings
    self.single_cluster_condition = single_cluster_condition
    self.single_cluster_affinity_threshold = single_cluster_affinity_threshold


class FallbackClusterer:
  """Fallback clusterer. So far we simply use AgglomerativeClustering."""

  def __init__(self, options):
    """Initilization of the fallback clusterer.

    Args:
      options: an object of FallbackOptions
    """
    self.options = options
    self.clusterer = AgglomerativeClustering()

  def predict(self, embeddings):
    return self.clusterer.fit_predict(embeddings)


def check_single_cluster(fallback_options, embeddings, affinity):
  """Check whether this is only a single cluster.

  This function is only called when min_clusters==1.

  Args:
    fallback_options: an object of FallbackOptions
    embeddings: numpy array of shape (n_samples, n_features)
    affinity: the affinity matrix of shape (n_samples, (n_samples)

  Returns:
    a boolean, where True means there is only a single cluster
  """
  if (fallback_options.single_cluster_condition ==
      SingleClusterCondition.AllAffinity):
    if (affinity.min() >
        fallback_options.single_cluster_affinity_threshold):
      return True
  elif (fallback_options.single_cluster_condition ==
        SingleClusterCondition.NeighborAffinity):
    neighbor_affinity = np.diag(affinity, k=1)
    if (neighbor_affinity.min() >
        fallback_options.single_cluster_affinity_threshold):
      return True
  elif (fallback_options.single_cluster_condition ==
        SingleClusterCondition.AffinityStd):
    if (np.std(affinity) <
        fallback_options.single_cluster_affinity_threshold):
      return True
  elif (fallback_options.single_cluster_condition ==
        SingleClusterCondition.AffinityGmmBic):
    gmm1 = GaussianMixture(n_components=1)
    gmm2 = GaussianMixture(n_components=2)
    affinity_values = np.expand_dims(affinity.flatten(), 1)
    gmm1.fit(affinity_values)
    gmm2.fit(affinity_values)
    bic1 = gmm1.bic(affinity_values)
    bic2 = gmm2.bic(affinity_values)
    return bic1 < bic2
  elif (fallback_options.single_cluster_condition ==
        SingleClusterCondition.FallbackClusterer):
    temp_clusterer = FallbackClusterer(fallback_options)
    temp_labels = temp_clusterer.predict(embeddings)
    if np.unique(temp_labels).size == 1:
      return True
  else:
    raise TypeError("Unsupported single_cluster_condition")
  return False
