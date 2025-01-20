import os
import sys
import tqdm
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample forge packs")
    parser.add_argument("--data-dir", type=str, help="The directory to get seventeen lands data from")
    parser.add_argument("--output-file", type=str, help="The file to output the p1 data to")
    args = parser.parse_args()

    # Prepare useful variables
    from seventeenlands_datasets import get_p1_data_ddf, configure_dask

    configure_dask()
    p1_ddf = get_p1_data_ddf(args.data_dir)
    # Render to pandas dataframe
    p1_df = p1_ddf.compute()

    columns = p1_df.columns

    # Get only the pack_card_* columns
    card_cols = columns[columns.str.startswith("pack_card_")]

    # grab only the card columns from p1_df
    p1_df = p1_df[card_cols]

    # Strip the "pack_card_" prefix from p1_df's columns
    p1_df.columns = p1_df.columns.str[len("pack_card_"):]

    # Save the p1 data to a csv file
    p1_df.to_csv(args.output_file, index=False)
