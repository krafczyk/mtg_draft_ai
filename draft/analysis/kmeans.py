from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from numpy.typing import ArrayLike, NDArray
from typing import Callable
import pandas as pd
import numpy as np
import numbers


def reorder_kmeans(km: KMeans,
                   key: Callable[[NDArray[np.float32]], np.float32]|None=None,
                   reverse: bool=False):
    if not hasattr(km, "cluster_centers_"):
        raise ValueError("KMeans estimator is not fitted (missing cluster_centers_).")

    centers = km.cluster_centers_
    if key is None:
        if centers.shape[-1] == 1:
            key = lambda c: c
        else:
            key = lambda c: c**2
    keys = np.array([key(c) for c in centers])

    # Stable sort to keep deterministic behavior on ties
    order = np.argsort(keys, kind="mergesort", axis=0).reshape(-1)
    if reverse:
        order = order[::-1]

    # inv[old_idx] = new_idx
    inv = np.empty_like(order)
    inv[order] = np.arange(order.size)

    # Apply reordering
    km.cluster_centers_ = centers[order]
    if hasattr(km, "labels_"):
        km.labels_ = km.labels_[order]


def compute_cluster_stats(km, X):
    labels = km.predict(X)
    stats = []
    for i, c in enumerate(km.cluster_centers_):
        xs = X[labels == i]
        mean = xs.mean()
        std = xs.std()
        stats.append({'center': c, 'mean': mean, 'std': std})
    return stats


def find_best_clusters(
        X: NDArray[np.float32],
        min_clusters=2,
        max_clusters=10,
        metric='silhouette',
        return_all=False,
        **mdl_kwargs):

    x = X.reshape(-1,1)

    if 'n_init' not in mdl_kwargs:
        mdl_kwargs['n_init'] = 'auto'
    if 'random_state' not in mdl_kwargs:
        mdl_kwargs['random_state'] = 0
    if 'n_clusters' in mdl_kwargs:
        raise ValueError("Can't set the cluster number like this.")

    fit_results = {}
    best_score = None
    best_res = None
    for k in range(min_clusters, max_clusters+1):
        km = KMeans(
            n_clusters=k, **mdl_kwargs)
        km.fit(X)
        reorder_kmeans(km)
        labels = km.predict(X)
        score = silhouette_score(X, labels)
        fit_results[k] = km
        if best_score is None:
            best_score = score
            best_res = fit_results[k]
        elif score > best_score: 
            best_score = score
            best_res = fit_results[k]

    if return_all:
        best_res, fit_results
    else:
        best_res
