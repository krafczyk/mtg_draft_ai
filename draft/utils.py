import dask


def configure_dask():
    _ = dask.config.set(scheduler='processes')
