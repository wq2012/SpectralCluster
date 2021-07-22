"""Utility functions."""

import numpy as np

EPS = 1e-10


def compute_affinity_matrix(embeddings):
  """Compute the affinity matrix from data.

  Note that the range of affinity is [0, 1].

  Args:
    embeddings: numpy array of shape (n_samples, n_features)

  Returns:
    affinity: numpy array of shape (n_samples, n_samples)
  """
  # Normalize the data.
  l2_norms = np.linalg.norm(embeddings, axis=1)
  embeddings_normalized = embeddings / l2_norms[:, None]
  # Compute cosine similarities. Range is [-1,1].
  cosine_similarities = np.matmul(embeddings_normalized,
                                  np.transpose(embeddings_normalized))
  # Compute the affinity. Range is [0,1].
  # Note that this step is not mentioned in the paper!
  affinity = (cosine_similarities + 1.0) / 2.0

  return affinity


def compute_laplacian(affinity, lp_type="graph_cut", eps=EPS):
  """Compute the Laplacian matrix.

  Args:
    affinity: the affinity matrix of input data
    lp_type: str. "unnormalized", "graph_cut", or "random_walk". if
      "unnormalized", compute the unnormalied laplacian. if "graph_cut", compute
      the graph cut view normalized laplacian, D^{-1/2}LD^{-1/2}. if
      "random_walk", compute the random walk view normalized laplacian, D^{-1}L
    eps: a small value for numerial stability

  Returns:
    laplacian: unnormalized graph laplacian
    laplacian_norm: normalized graph laplacian
  """
  degree = np.diag(np.sum(affinity, axis=1))
  laplacian = degree - affinity
  if lp_type == "unnormalized":
    return laplacian
  elif lp_type == "random_walk":
    # Random walk normalized version
    degree_norm = np.diag(1 / (np.diag(degree) + eps))
    laplacian_norm = degree_norm.dot(laplacian)
    return laplacian_norm
  elif lp_type == "graph_cut":
    # Graph cut normalized version
    degree_norm = np.diag(1 / (np.sqrt(np.diag(degree)) + eps))
    laplacian_norm = degree_norm.dot(laplacian).dot(degree_norm)
    return laplacian_norm
  else:
    raise ValueError(
        "The lp_type should be 'unnormalized', 'random_walk', or 'graph_cut'.")


def compute_sorted_eigenvectors(input_matrix, descend=True):
  """Sort eigenvectors by the real part of eigenvalues.

  Args:
    input_matrix: the matrix to perform eigen analysis with shape (M, M)
    descend: sort eigenvalues in a descending order. Default is True

  Returns:
    w: sorted eigenvalues of shape (M,)
    v: sorted eigenvectors, where v[;, i] corresponds to ith largest
       eigenvalue
  """
  # Eigen decomposition.
  eigenvalues, eigenvectors = np.linalg.eig(input_matrix)
  eigenvalues = eigenvalues.real
  eigenvectors = eigenvectors.real
  if descend:
    # Sort from largest to smallest.
    index_array = np.argsort(-eigenvalues)
  else:
    # Sort from smallest to largest.
    index_array = np.argsort(eigenvalues)
  # Re-order.
  w = eigenvalues[index_array]
  v = eigenvectors[:, index_array]
  return w, v


def compute_number_of_clusters(eigenvalues,
                               max_clusters=None,
                               stop_eigenvalue=1e-2,
                               descend=True,
                               eps=EPS):
  """Compute number of clusters using EigenGap principle.

  Use maximum EigenGap principle to find the number of clusters. The eigenvalues
  and eigenvectors are sorted in a descending order.

  Args:
    eigenvalues: sorted eigenvalues of the affinity matrix
    max_clusters: max number of clusters allowed
    stop_eigenvalue: we do not look at eigen values smaller than this
    descend: sort eigenvalues in a descending order. Default is True
    eps: a small value for numerial stability

  Returns:
    number of clusters as an integer
  """
  max_delta = 0
  max_delta_index = 0
  range_end = len(eigenvalues)
  if max_clusters and max_clusters + 1 < range_end:
    range_end = max_clusters + 1

  if not descend:
    # The first eigen value is always 0 in an ascending order
    for i in range(1, range_end - 1):
      delta = eigenvalues[i + 1] / (eigenvalues[i] + eps)
      if delta > max_delta:
        max_delta = delta
        max_delta_index = i + 1  # Index i means i+1 clusters
    return max_delta_index
  else:
    for i in range(1, range_end):
      if eigenvalues[i - 1] < stop_eigenvalue:
        break
      delta = eigenvalues[i - 1] / eigenvalues[i]
      if delta > max_delta:
        max_delta = delta
        max_delta_index = i
    return max_delta_index


def enforce_ordered_labels(labels):
  """Transform the label sequence to an ordered form.

  This is the same type of label sequence used in the paper "Discriminative
  neural clustering for speaker diarisation". This makes the label sequence
  permutation invariant.

  Args:
    labels: an array of integers

  Returns:
    new_labels: an array of integers, where it starts with 0 and smaller
      labels always appear first
  """
  new_labels = labels.copy()
  max_label = -1
  label_map = {}
  for element in labels.tolist():
    if element not in label_map:
      max_label += 1
      label_map[element] = max_label
  for key in label_map:
    new_labels[labels == key] = label_map[key]
  return new_labels
