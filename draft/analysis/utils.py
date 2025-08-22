def get_cluster(mdl, i, x_series):
    labels = mdl.predict(x_series.to_numpy().reshape(-1,1))
    return x_series[labels == i]
