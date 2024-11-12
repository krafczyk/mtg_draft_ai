#!/bin/bash

# Check if input file is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <file.csv.gz>"
  exit 1
fi

# Input file and output directory
input_file="$1"
filename=$(basename "$input_file" .csv.gz)
output_dir="$filename"

# Create output directory if it doesn't exist
mkdir -p "$output_dir"

# Extract the header line
header=$(gzip -dc "$input_file" | head -n 1)

# Decompress, skip the header, and split the file into 1 MB chunks
gzip -dc "$input_file" | tail -n +2 | split -b 104857600 - "$output_dir/${filename}_chunk_"

# Rename chunks to have .csv extension
for chunk in "$output_dir/${filename}_chunk_"*; do
  (echo "$header" && cat "$chunk") > "${chunk}.csv"
  rm "$chunk"
done

echo "Chunks created in directory: $output_dir with .csv extensions"
