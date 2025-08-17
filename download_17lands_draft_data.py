import subprocess
import os
from typing import cast

def download_set_draft_data(set_code: str, draft_type: str):
    draft_data = f"draft_data_public.{set_code}.{draft_type}.csv.gz"
    if not os.path.exists(draft_data):
        url = f"https://17lands-public.s3.amazonaws.com/analysis_data/draft_data/{draft_data}"
        print(f"Downloading {draft_data} from {url}...")
        subprocess.run(["wget", url], check=True)
    else:
        print(f"{draft_data} already exists, skipping download.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download 17Lands draft data.")
    parser.add_argument("--set-code", type=str, required=True, help="The set code to fetch")
    args = parser.parse_args()

    set_code = cast(str, args.set_code)

    download_set_draft_data(set_code, "PremierDraft")
    download_set_draft_data(set_code, "TradDraft")
