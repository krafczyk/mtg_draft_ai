#!/bin/bash

dirpath=$(dirname "$0")

for file in $(find . -path "*17lands_data*" -type f -name "*.csv.gz"); do
    # Call the Python script with the set code
    target_dir=${file%.csv.gz}
    if [ ! -d "$target_dir" ]; then
        python "${dirpath}/split_csv_gz.py" "$file"
    fi
done
