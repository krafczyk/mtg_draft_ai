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
    output_filepath="pack_data/pack_data_${set_code}.csv"
    if [ ! -d "${output_filepath}" ]; then
        echo "Extracting pack data for set $set_code"
        python "${dirpath}/extract_pack_data_from_draft_data_17lands.py" --draft-data $(find . -maxdepth 2 -mindepth 2 -path "*17lands_data*" -type d -name "*${set_code}*") --output "${output_filepath}" --num-workers 4
    fi;
done
