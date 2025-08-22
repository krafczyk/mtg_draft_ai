import pandas as pd

def get_pack_data(filepath:str):
    return pd.read_csv(filepath)
