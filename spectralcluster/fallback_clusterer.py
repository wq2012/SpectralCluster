"""In some cases, we use a fallback clusterer instead of spectral.

Spectral clustering exploits the global structure of the data. But there are
cases where spectral clustering does not work as well as some other simpler
clustering methods, such as when the number of embeddings is too small.
"""

import enum
from sklearn.cluster import AgglomerativeClustering


class SingleClusterCondition(enum.Enum):
  """Which condition do we use for deciding single-vs-multi cluster(s)."""
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
               single_cluster_condition=SingleClusterCondition.AllAffinity,
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
