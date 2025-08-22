import dask.dataframe as dd
from distributed.utils_comm import pack_data
from replay_dtypes import get_dtypes
import os
import glob
import argparse
from typing import cast


def get_draft_data(csv_paths: list[str]):
    dtypes = get_dtypes(filename=csv_paths[0])
    return dd.read_csv(csv_paths, dtype=dtypes)

def extract_pack_data(draft_df):
    test_df = draft_df.head(n=10)
    cols = test_df.columns
    pack_cols = cols.to_series().str.contains("pack_card_")
    return draft_df.loc[draft_df['pick_number'] == 0, pack_cols]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract pack data from 17Lands draft data.")
    _ = parser.add_argument("--draft-data", required=True, nargs="+", type=str, help="A list of directories containing 17lands draft data .csv files or 17lands draft data csvs.")
    _ = parser.add_argument("--output", required=True, type=str, help="The file to use for output")
    _ = parser.add_argument("--num-workers", default=8, type=int, help="The number of work processes to use")
    args = parser.parse_args()

    data_items = cast(list[str], args.draft_data)
    output_filepath = cast(str, args.output)
    num_workers = cast(int, args.num_workers)

    csv_list = []
    for data_item in data_items:
        if os.path.isdir(data_item):
            csv_list += glob.glob(os.path.join(data_item, "*.csv"))
        elif os.path.isfile(data_item) and data_item.endswith(".csv"):
            csv_list.append(data_item)
        else:
            raise ValueError(f"The path {data_item} is neither a directory nor a .csv file.")

    # Loading the draft data
    draft_ddf = get_draft_data(csv_list)
    print(f"Building extract pack data definition from draft data...")
    pack_data_ddf = extract_pack_data(draft_ddf)
    print(f"Extracting pack data definition from draft data...")
    pack_data_df = pack_data_ddf.compute(scheduler="processes", num_workers=num_workers)
    print(f"Renaming columns")
    pack_data_df = pack_data_df.rename(columns=lambda x: x.replace("pack_card_", ""))
    print(f"Saving pack data to {output_filepath}...")
    pack_data_df.to_csv(output_filepath, index=False)
