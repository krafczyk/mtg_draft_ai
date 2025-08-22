import sklearn
import numpy as np
from numpy.typing import NDArray
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go


def find_best_clusters(
        X: NDArray[np.float32],
        min_clusters=1,
        max_clusters=10,
        metric='bic',
        return_all=False,):

    x = X.reshape(-1,1)

    fit_results = {}
    best_fit_metric = None
    best_fit_gmm = None

    for n_clusters in range(1,max_clusters+1):
        gmm = sklearn.mixture.GaussianMixture(
            n_components=n_clusters,
            random_state=0)
        gmm.fit(x)

        fit_results[n_clusters] = gmm

        metric = None
        if metric == 'bic':
            metric = gmm.bic(x)
        else:
            raise ValueError("Unsupported Metric")

        if best_fit_metric is None:
            best_fit_gmm = gmm
            best_fit_metric = metric
        elif metric < best_fit_metric:
            best_fit_gmm = gmm
            best_fit_metric = metric

    if return_all:
        return best_fit_gmm, fit_results
    else:
        return best_fit_gmm
