import dask
import dask.dataframe as ddf
import os
from replay_dtypes import get_dtypes

def configure_dask():
    _ = dask.config.set(scheduler='processes')

def get_data_dir_dtypes(data_dir: str) -> dict:
    dir_content = os.listdir(data_dir)
    # Find all .csv files
    dir_content = [d for d in dir_content if d.endswith(".csv")]
    # Lexographically sort the files
    dir_content.sort()
    first_file = dir_content[0]
    # Get dtypes
    return get_dtypes(filename=os.path.join(data_dir, first_file))

#def get_data_dir_columns(data_dir: str) -> list:
#    dtypes = get_data_dir_dtypes(data_dir)
    

def get_p1_data_ddf(data_dir: str) -> ddf.DataFrame:
    dtypes = get_data_dir_dtypes(data_dir)
    # Load the dataset
    draft_data_ddf = ddf.read_csv(os.path.join(data_dir, "*.csv"), dtype=dtypes)

    # Get only data for pack 1 pick 1
    return draft_data_ddf[draft_data_ddf['pick_number'] == 0]
