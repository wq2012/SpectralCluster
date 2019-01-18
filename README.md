# Spectral Clustering

## Overview

This is a Python re-implementation of the spectral clustering algorithm in the
paper [Speaker Diarization with LSTM](https://google.github.io/speaker-id/publications/LstmDiarization/).

## Disclaimer

**This is not the original implementation used by the paper.**

Specifically, in this implementation, we use the K-Means from
[scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html),
which does NOT support customized distance measure like cosine distance.
