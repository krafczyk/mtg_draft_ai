import sklearn
import numpy as np
from numpy.typing import NDArray
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go


def _variances_1d_from_gmm(gmm: sklearn.mixture.GaussianMixture) -> NDArray:
    """Extract per-component variances for 1-D data, regardless of covariance_type."""
    ct = gmm.covariance_type
    if ct == "full":        # shape: (k, 1, 1)
        var = gmm.covariances_[:, 0, 0]
    elif ct == "tied":      # shape: (1, 1)
        var = np.repeat(gmm.covariances_[0, 0], gmm.n_components)
    elif ct == "diag":      # shape: (k, 1)
        var = gmm.covariances_[:, 0]
    elif ct == "spherical": # shape: (k,)
        var = gmm.covariances_
    else:
        raise ValueError(f"Unsupported covariance_type: {ct}")
    return var


def plot_gmm_1d_plotly(x, gmm: sklearn.mixture.GaussianMixture, bins="auto", density=True, hist_opacity=0.35):
    """
    Plotly figure: histogram of x with per-component Gaussians from a fitted GaussianMixture.
    Each component curve is truncated to mean ± 2σ and scaled by its mixture weight.
    If density=True, the histogram is a density and curves are true densities.
    If density=False, curves are scaled to counts (N * mean_bin_width) for visual comparability.
    """
    x = np.asarray(x).ravel()
    N = len(x)

    # Compute bin edges first so we know the mean bin width (for count scaling)
    # Use numpy to materialize the edges even when bins="auto"
    counts_np, edges = np.histogram(x, bins=bins, density=False)
    bin_width = np.diff(edges).mean() if len(edges) > 1 else 1.0

    fig = go.Figure()

    # Histogram trace (Plotly computes bins too; we match density via histnorm)
    histnorm = "probability density" if density else None
    fig.add_trace(go.Histogram(
        x=x,
        nbinsx=None if isinstance(bins, str) else bins,
        histnorm=histnorm,
        opacity=hist_opacity,
        marker=dict(line=dict(width=1)),
        name="Data"
    ))

    # Extract GMM params
    means = gmm.means_.ravel()
    variances = _variances_1d_from_gmm(gmm)
    stds = np.sqrt(variances)
    weights = gmm.weights_.ravel()

    # Sort components by mean for nice ordering
    order = np.argsort(means)
    means, stds, weights = means[order], stds[order], weights[order]

    # Add each component curve (truncated to ±2σ) and a vertical dotted mean line
    for idx, (m, s, w) in enumerate(zip(means, stds, weights), start=1):
        if s <= 0:
            continue
        xs = np.linspace(m - 2*s, m + 2*s, 400)
        pdf = np.exp(-0.5 * ((xs - m) / s)**2) / (s * np.sqrt(2*np.pi))

        if density:
            ys = w * pdf
        else:
            ys = w * pdf * N * bin_width  # convert density to counts for comparison to histogram

        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="lines",
            line=dict(width=2),
            name=f"Comp {idx}"
        ))
        # Dotted vertical mean
        fig.add_trace(go.Scatter(
            x=[m, m],
            y=[0, (w * (1/(s*np.sqrt(2*np.pi))) if density else (w * (1/(s*np.sqrt(2*np.pi))) * N * bin_width))],
            mode="lines",
            line=dict(width=1, dash="dot"),
            showlegend=False
        ))

    fig.update_layout(
        title="GMM components over data histogram (curves truncated to ±2σ)",
        xaxis_title="x",
        yaxis_title="Density" if density else "Count",
        barmode="overlay",
        template="plotly_white"
    )
    # Overlay bars and lines
    fig.update_traces(marker_line_color="black", selector=dict(type="histogram"))
    return fig
