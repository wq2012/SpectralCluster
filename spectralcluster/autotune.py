"""Auto-tuning hyper-parameters."""

import enum
import numpy as np
import typing

MIN_SEARCH_STEP = 1e-04


class AutoTuneProxy(enum.Enum):
  """What proxy to use as the auto-tuning target."""

  # The original proxy used in:
  # Park, Tae Jin, et al. "Auto-tuning spectral clustering for speaker
  # diarization using normalized maximum eigengap." IEEE Signal Processing
  # Letter 2019.
  PercentileOverNME = enum.auto()

  # The modified proxy used in:
  # Xia, Wei, et al. "Turn-to-diarize: Online speaker diarization constrained
  # by transformer transducer speaker turn detection." ICASSP 2022.
  # https://arxiv.org/abs/2109.11641
  PercentileSqrtOverNME = enum.auto()


class AutoTune:
  """AutoTune Class.

  This auto-tuning method is implemented based on this paper:
  Park, Tae Jin, et al. "Auto-tuning spectral clustering for speaker
  diarization using normalized maximum eigengap." IEEE Signal Processing Letter
  2019.
  """

  def __init__(self,
               p_percentile_min: float = 0.60,
               p_percentile_max: float = 0.95,
               init_search_step: float = 0.01,
               search_level: int = 1,
               proxy: AutoTuneProxy = AutoTuneProxy.PercentileSqrtOverNME):
    """Initialization of the autotune arguments.

    Args:
      p_percentile_min: minimum value of p_percentile
      p_percentile_max: maximum value of p_percentile
      init_search_step: initial search step size for auto-tuning
      search_level: hierarchical search level for auto-tuning
      proxy: which proxy to minimize for auto-tuning
    """
    self.p_percentile_min = p_percentile_min
    self.p_percentile_max = p_percentile_max
    self.search_step = init_search_step
    self.search_level = search_level
    if not isinstance(proxy, AutoTuneProxy):
      raise TypeError("proxy must be an instance of AutoTuneProxy")
    self.proxy = proxy

  def get_percentile_range(self) -> typing.Sequence[float]:
    """Get the current percentile search range."""
    num_steps = int(
        np.ceil(
            (self.p_percentile_max - self.p_percentile_min) / self.search_step))
    return list(
        np.linspace(self.p_percentile_min, self.p_percentile_max, num_steps))

  def update_percentile_range(self,
                              p_percentile_min: float,
                              p_percentile_max: float,
                              search_step: float) -> typing.Sequence[float]:
    """Update the percentile search range."""
    self.p_percentile_min = p_percentile_min
    self.p_percentile_max = p_percentile_max
    self.search_step = search_step
    return self.get_percentile_range()

  def tune(self, p_percentile_to_ratio: typing.Callable) -> (
      typing.Tuple[np.ndarray, int, float]):
    """Tune the hyper-parameter p_percentile.

    Use a proxy ratio of DER to tune the hyper-parameter p_percentile. It also
    performs some side work to do affinity refinement, eigen decomposition, and
    estimate the number of clusters.

    Args:
      p_percentile_to_ratio: a callable to compute the `ratio` given a
        `p_percentile` value

    Returns:
      eigenvectors: sorted eigenvectors. numpy array of shape
      (n_samples, n_samples)
      n_clusters: number of clusters as an integer
      best_p_percentile: p_percentile value that minimizes the ratio
    """
    p_percentile_range = self.get_percentile_range()
    searched = dict()
    for _ in range(self.search_level):
      min_ratio = np.inf
      for index, p_percentile in enumerate(p_percentile_range):
        if p_percentile in searched:
          continue
        # ratio is a proxy value of DER. We minimize this ratio
        # to find a good p_percentile
        ratio, eigenvectors_p, n_clusters_p = p_percentile_to_ratio(
            p_percentile)
        searched[p_percentile] = ratio
        if ratio < min_ratio:
          min_ratio = ratio
          eigenvectors = eigenvectors_p
          n_clusters = n_clusters_p
          best_p_percentile = p_percentile
          best_p_percentile_index = index
      # If the search range is not valid or search step is too small, we stop
      if not p_percentile_range or len(
          p_percentile_range) == 1 or self.search_step < MIN_SEARCH_STEP:
        break
      # Update the search range of p_percentile.
      # We search again from `start_index` position to `end_index` position
      # which is `local_search_dist` away from the found
      # `best_p_percentile_index` position. `search_step` is reduced to half of
      # the original size
      local_search_dist = max(2, len(p_percentile_range) // 8)
      start_index = max(0, best_p_percentile_index - local_search_dist)
      end_index = min(
          len(p_percentile_range) - 1,
          best_p_percentile_index + local_search_dist)
      p_percentile_min = p_percentile_range[start_index]
      p_percentile_max = p_percentile_range[end_index]
      self.search_step = self.search_step / 2
      p_percentile_range = self.update_percentile_range(p_percentile_min,
                                                        p_percentile_max,
                                                        self.search_step)
    return eigenvectors, n_clusters, best_p_percentile
