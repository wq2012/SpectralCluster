# Spectral Clustering

## Overview

This is a Python re-implementation of the spectral clustering algorithm in the
paper [Speaker Diarization with LSTM](https://google.github.io/speaker-id/publications/LstmDiarization/).

![refinement](https://raw.githubusercontent.com/wq2012/SpectralCluster/master/resources/refinement.png)

## Disclaimer

**This is not the original implementation used by the paper.**

Specifically, in this implementation, we use the K-Means from
[scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html),
which does NOT support customized distance measure like cosine distance.

## Tutorial

Simply use the `cluster()` method of class `SpectralClusterer` to perform
spectral clustering.

```python
from spectralcluster import SpectralClusterer

clusterer = SpectralClusterer(
    min_clusters=2,
    max_clusters=100,
    p_percentile=0.95,
    gaussian_blur_sigma=1)

labels = clusterer.cluster(X)
```

The input `X` is a numpy array of shape `(n_samples, n_features)`,
and the returned `1abels` is a numpy array of shape `(n_samples,)`.

For the complete list of parameters of the clusterer, see
`spectralcluster/spectral_clusterer.py`.