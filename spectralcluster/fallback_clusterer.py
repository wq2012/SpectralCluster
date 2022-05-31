"""In some cases, we use a fallback clusterer instead of spectral.

Spectral clustering exploits the global structure of the data. But there are
cases where spectral clustering does not work as well as some other simpler
clustering methods, such as when the number of embeddings is too small.
"""

from sklearn.cluster import AgglomerativeClustering


class FallbackOptions:
  """Options for fallback options."""

  def __init__(self, spectral_min_embeddings=1):
    """Initialization of the fallback clusterer.

    Args:
      spectral_min_embeddings: we only run spectral clusterer if we have at
        least these many embeddings; otherwise we run fallback clusterer
    """
    self.spectral_min_embeddings = spectral_min_embeddings


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
