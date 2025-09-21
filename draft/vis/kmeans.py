import numpy as np


def hvplot_kmeans_1d_overlay(x, cluster_stats, bins="auto", density=True,
                             band_type="std", hist_opacity=0.35, title=None):
    """
    band_type: "std" -> shade +/- 1 std around center (from data within cluster)
               "voronoi" -> shade the Voronoi cell (midpoints to neighboring centers)
    """
    import numpy as np
    import holoviews as hv
    import hvplot.pandas  # registers .hvplot and sets up Bokeh backend

    x = np.asarray(x).ravel()

    # Build histogram with numpy so we can support bins="auto" exactly.
    counts_np, edges = np.histogram(x, bins=bins, density=density)
    vd = "Density" if density else "Count"

    # Histogram element
    hist = hv.Histogram(
        (edges, counts_np),
        kdims=[hv.Dimension("x")],
        vdims=[hv.Dimension(vd)],
    ).opts(
        fill_alpha=hist_opacity,
        line_color="black",
        show_legend=False,
        tools=["hover"],
    )

    # Extract 1D centers (norm for multi-d)
    if cluster_stats[0]["center"].size == 1:
        centers = [float(s["center"][0]) for s in cluster_stats]
    else:
        centers = [float(np.linalg.norm(s["center"])) for s in cluster_stats]

    # Voronoi cell boundaries (assuming input order corresponds to neighbor order)
    left_bounds, right_bounds = [], []
    for i, c in enumerate(centers):
        left = (-np.inf if i == 0 else 0.5 * (centers[i - 1] + c))
        right = (np.inf if i == len(centers) - 1 else 0.5 * (c + centers[i + 1]))
        left_bounds.append(left)
        right_bounds.append(right)

    xmin, xmax = float(np.min(x)), float(np.max(x))

    spans = []
    vlines = []
    for i, c in enumerate(centers):
        std = float(cluster_stats[i]["std"])
        if band_type == "std":
            lo, hi = c - std, c + std
        elif band_type == "voronoi":
            lo = max(left_bounds[i], xmin)
            hi = min(right_bounds[i], xmax)
        else:
            raise ValueError("band_type must be 'std' or 'voronoi'")

        if not np.isfinite(lo):
            lo = xmin
        if not np.isfinite(hi):
            hi = xmax

        spans.append(
            hv.VSpan(lo, hi).opts(color="black", line_alpha=0.0, fill_alpha=0.08, show_legend=False)
        )
        vlines.append(
            hv.VLine(c).opts(line_dash="dotted", line_width=1, color="black", show_legend=False)
        )

    overlay = hist
    if spans:
        overlay = overlay * hv.Overlay(spans)
    if vlines:
        overlay = overlay * hv.Overlay(vlines)

    overlay = overlay.opts(
        title=title or f"K-Means centers with {'±1σ' if band_type=='std' else 'Voronoi'} bands",
        xlabel="x",
        ylabel=("Density" if density else "Count"),
    )
    return overlay


def plotly_kmeans_1d_overlay(x, cluster_stats, bins="auto", density=True,
                           band_type="std", hist_opacity=0.35, title=None):
    """
    band_type: "std" -> shade +/- 1 std around center (from data within cluster)
               "voronoi" -> shade the Voronoi cell (midpoints to neighboring centers)
    """
    import plotly.graph_objects as go
    x = np.asarray(x).ravel()
    N = len(x)

    # For count scaling if needed
    counts_np, edges = np.histogram(x, bins=bins, density=False)
    bin_width = np.diff(edges).mean() if len(edges) > 1 else 1.0

    fig = go.Figure()
    histnorm = "probability density" if density else None
    fig.add_trace(go.Histogram(
        x=x,
        nbinsx=None if isinstance(bins, str) else bins,
        histnorm=histnorm,
        opacity=hist_opacity,
        marker=dict(line=dict(width=1)),
        name="Data"
    ))

    if cluster_stats[0]['center'].size == 1:
        centers = list(map(lambda s: s['center'][0], cluster_stats))
    else:
        centers = list(map(lambda s: np.linalg.norm(s['center']), cluster_stats))

    # Voronoi cell boundaries (for optional shading)
    left_bounds, right_bounds = [], []
    for i, c in enumerate(centers):
        if i == 0:
            left = -np.inf
        else:
            left = 0.5 * (centers[i-1] + c)
        if i == len(centers) - 1:
            right = np.inf
        else:
            right = 0.5 * (c + centers[i+1])
        left_bounds.append(left)
        right_bounds.append(right)

    # Add shaded bands and dotted means
    y_max_guess = None
    if density:
        # height around mean if it helps scale the mean line to visible range;
        # for kmeans (not parametric), just estimate using histogram peak
        y_max_guess = (counts_np.max() / N) / bin_width if np.any(counts_np) else 1.0
    else:
        y_max_guess = counts_np.max() if np.any(counts_np) else 1.0

    for i, c in enumerate(centers):
        std = cluster_stats[i]["std"]
        # Band extents
        if band_type == "std":
            lo, hi = c - std, c + std
        elif band_type == "voronoi":
            # clamp to data range for nicer view
            lo = max(left_bounds[i], x.min())
            hi = min(right_bounds[i], x.max())
        else:
            raise ValueError("band_type must be 'std' or 'voronoi'")

        # Shade as a vertical rectangle
        fig.add_shape(
            type="rect",
            x0=lo, x1=hi,
            y0=0, y1=y_max_guess,
            line=dict(width=0),
            fillcolor="rgba(0,0,0,0.08)",
            layer="below"
        )

       
        # Dotted vertical line at center
        fig.add_trace(go.Scatter(
            x=[c, c],
            y=[0, y_max_guess],
            mode="lines",
            line=dict(width=1, dash="dot"),
            name=f"Center {i+1}",
            hovertemplate=f"Center {i+1}: {c:.6g}<extra></extra>"
        ))

    fig.update_layout(
        title=title or f"K-Means centers with {'±1σ' if band_type=='std' else 'Voronoi'} bands",
        xaxis_title="x",
        yaxis_title="Density" if density else "Count",
        barmode="overlay",
        template="plotly_white",
        showlegend=False
    )
    # Make histogram bars outlined
    fig.update_traces(marker_line_color="black", selector=dict(type="histogram"))
    return fig
