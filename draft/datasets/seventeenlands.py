import pandas as pd
import dask.dataframe as ddf
import os
from replay_dtypes import get_dtypes
import numpy as np
from scipy import sparse


def get_data_dir_dtypes(data_dir: str) -> dict[str,str]:
    dir_content = os.listdir(data_dir)
    # Find all .csv files
    dir_content = [d for d in dir_content if d.endswith(".csv")]
    # Lexographically sort the files
    dir_content.sort()
    first_file = dir_content[0]
    # Get dtypes
    return get_dtypes(filename=os.path.join(data_dir, first_file))


def get_data_ddf(data_dir: str) -> ddf.DataFrame:
    dtypes = get_data_dir_dtypes(data_dir)
    # Load the dataset
    return ddf.read_csv(os.path.join(data_dir, "*.csv"), dtype=dtypes)


def get_p1_data_ddf(data_dir: str) -> ddf.DataFrame:
    draft_data_ddf = get_data_ddf(data_dir)

    # Get only data for pack 1 pick 1
    return draft_data_ddf[draft_data_ddf['pick_number'] == 0]

def get_card_df(filepath="cards.csv"):
    return pd.read_csv(filepath)


def get_pack_data_summary_sparse(filepath:str) -> pd.DataFrame:
    onehot_cols = None   # set explicitly if you know them

    rows_all, cols_all, data_all = [], [], []
    row_offset = 0

    for chunk in pd.read_csv(filepath, chunksize=10_000):
        if onehot_cols is None:
            # e.g., everything except these are one-hots:
            onehot_cols = [c for c in chunk.columns if c not in ("pack_id","label")]

        X = chunk[onehot_cols].to_numpy()
        r, c = np.nonzero(X)                    # positions of non-zeros in this chunk
        rows_all.append(r + row_offset)
        cols_all.append(c)
        data_all.append(X[r, c].astype(np.int32))  # 1s (or counts, if present)
        row_offset += X.shape[0]

    rows = np.concatenate(rows_all)
    cols = np.concatenate(cols_all)
    data = np.concatenate(data_all)

    csr = sparse.csr_matrix((data, (rows, cols)), shape=(row_offset, len(onehot_cols)))
    #csr.sum_duplicates()

    # Convert to pandas sparse with original column names
    return pd.DataFrame.sparse.from_spmatrix(csr, columns=onehot_cols)


def get_pack_data_summary_dense(filepath:str) -> pd.DataFrame:
    sparse_df = get_pack_data_summary_sparse(filepath)
    return sparse_df.sparse.to_dense()
