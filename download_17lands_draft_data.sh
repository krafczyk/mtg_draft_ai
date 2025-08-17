#!/bin/bash

dirpath=$(dirname "$0")

set_codes=(
    "TDM"
    "DFT"
    "PIO"
    "FDN"
    "DSK"
    "BLB"
    "MH3"
    "OTJ"
    "MKM"
    "KTK"
    "LCI"
    "WOE"
    "LTR"
    "MOM"
    "SIR"
    "ONE"
    "BRO"
    "DMU"
    "HBG"
    "SNC"
    "NEO"
    "VOW"
)

for set_code in "${set_codes[@]}"; do
    echo "Downloading data for set code: $set_code"
    # Call the Python script with the set code
    python "${dirpath}/download_17lands_draft_data.py" --set-code "$set_code"
done
