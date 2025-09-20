import numpy as np
import pandas as pd
from scipy import sparse


def get_pack_data_sparse(filepath:str) -> pd.DataFrame:
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
        data_all.append(X[r, c].astype(np.uint8))  # 1s (or counts, if present)
        row_offset += X.shape[0]

    rows = np.concatenate(rows_all)
    cols = np.concatenate(cols_all)
    data = np.concatenate(data_all)

    csr = sparse.csr_matrix((data, (rows, cols)), shape=(row_offset, len(onehot_cols)))
    csr.sum_duplicates()

    # Convert to pandas sparse with original column names
    return pd.DataFrame.sparse.from_spmatrix(csr, columns=onehot_cols)


def get_pack_data_sparse(filepath:str) -> pd.DataFrame:
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
        data_all.append(X[r, c].astype(np.uint8))  # 1s (or counts, if present)
        row_offset += X.shape[0]

    rows = np.concatenate(rows_all)
    cols = np.concatenate(cols_all)
    data = np.concatenate(data_all)

    csr = sparse.csr_matrix((data, (rows, cols)), shape=(row_offset, len(onehot_cols)))
    csr.sum_duplicates()

    # Convert to pandas sparse with original column names
    return pd.DataFrame.sparse.from_spmatrix(csr, columns=onehot_cols)


def get_pack_data_dense(filepath:str) -> pd.DataFrame:
    sparse_df = get_pack_data_sparse(filepath)
    return sparse_df.sparse.to_dense()


def java_pack_list_to_sparse_df(java_packs, cards):
    import jpype
    # Build vocab
    card_to_idx = {card: idx for idx, card in enumerate(cards)}
    HashMap = jpype.JClass("java.util.HashMap")
    j_index = HashMap()
    for k, v in card_to_idx.items():
        j_index.put(k, jpype.types.JInt(v))

    # Build sparse matrix
    from com.krafczyk.forge import PackFns
    csr_res = PackFns.packsToCsr(
        java_packs,
        j_index,
        jpype.types.JInt(len(cards)))
    indptr = np.asarray(csr_res.indptr, dtype=np.int32, copy=True)
    indices = np.asarray(csr_res.indices, dtype=np.int32, copy=True)
    data_i8 = np.asarray(csr_res.data, dtype=np.int8, copy=True)
    data_i32 = data_i8.astype(np.int32, copy=False)   # widen
    A = sparse.csr_matrix((data_i32, indices, indptr), shape=(indptr.size-1, len(cards)))
    #A.sum_duplicates()
    return pd.DataFrame.sparse.from_spmatrix(A, columns=cards)


def java_pack_list_to_dense_df(java_packs, cards):
    import jpype
    # Build vocab
    card_to_idx = {card: idx for idx, card in enumerate(cards)}
    HashMap = jpype.JClass("java.util.HashMap")
    j_index = HashMap()
    for k, v in card_to_idx.items():
        j_index.put(k, jpype.types.JInt(v))

    from com.krafczyk.forge import PackFns
    dense_java = PackFns.packsToDense(java_packs, j_index, jpype.types.JInt(len(cards)))

    # Convert to NumPy and DataFrame
    dense = np.asarray(dense_java, dtype=np.int8, copy=True).view(np.uint8)  # byte[][] -> uint8
    return pd.DataFrame(dense, columns=cards)
