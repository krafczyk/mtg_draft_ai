from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from numpy.typing import ArrayLike
from typing import Callable
import pandas as pd
import numpy as np
import numbers


def find_best_kmeans_silhouette(data: ArrayLike | pd.DataFrame, k_min:int=2, k_max:int=10):
    best_score = -1
    best_kmeans = None
    
    for k in range(k_min, k_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=42).fit(data)
        labels = kmeans.labels_
        score = silhouette_score(data, labels)
        
        if score > best_score:
            best_score = score
            best_kmeans = kmeans
    return best_kmeans


def sort_kmeans_clusters(
    kmeans: KMeans,
    key: None | Callable[[ArrayLike], numbers.Real]=None,
    reverse:bool=True) -> None:

    """
    Sorts the clusters in a fitted KMeans model based on a given criterion.

    Parameters
    ----------
    kmeans : KMeans object
        A fitted KMeans model.
        
    key : callable, optional
        A function that takes a cluster center (1D array) and returns a sortable value.
        Default is sum of the cluster coordinates.
        
    reverse : bool, optional
        Whether to sort in descending order (True) or ascending order (False).
        Default is True (descending order).
    
    Returns
    -------
    None
        The function updates kmeans.cluster_centers_ and kmeans.labels_ in place.
    """
    if not hasattr(kmeans, 'cluster_centers_'):
        raise ValueError("The KMeans model must be fitted before sorting.")
    
    if key is None:
        # Default key: sum of cluster coordinates
        key = lambda c: np.sum(c)
    
    # Compute the key value for each cluster center
    cluster_keys = np.array([key(center) for center in kmeans.cluster_centers_])
    
    # Determine new order of clusters
    sort_order = np.argsort(cluster_keys)
    if reverse:
        sort_order = sort_order[::-1]
    
    # Reorder the cluster centers
    kmeans.cluster_centers_ = kmeans.cluster_centers_[sort_order]
    
    # Create a label mapping (old_label -> new_label)
    label_map = np.empty_like(sort_order)
    for new_label, old_label in enumerate(sort_order):
        label_map[old_label] = new_label
    
    # Apply the mapping to labels_
    kmeans.labels_ = label_map[kmeans.labels_]
