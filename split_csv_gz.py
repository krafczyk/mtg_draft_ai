import gzip
import os
import multiprocessing
import argparse

def split_csv_by_size(input_file, max_chunk_size=50 * 1024 * 1024):  # Default 50 MB
    # Define output directory based on input file name (stem without .csv.gz)
    base_dir = os.path.dirname(input_file)
    filename_stem = os.path.basename(input_file).replace('.csv.gz', '')
    output_dir = os.path.join(base_dir, filename_stem)
    os.makedirs(output_dir, exist_ok=True)

    # Open the gzip file and read the header
    with gzip.open(input_file, 'rt') as f:
        header = f.readline().strip()  # Read the header line

    # Function to write each chunk with the header
    def write_chunk(data_chunk, chunk_num):
        chunk_filename = os.path.join(output_dir, f"{filename_stem}_chunk_{chunk_num}.csv")
        with open(chunk_filename, 'w') as chunk_file:
            chunk_file.write(header + '\n')  # Write the header
            chunk_file.write(data_chunk)  # Write chunk data

    # Process the file in size-limited chunks
    chunk_num = 0
    with gzip.open(input_file, 'rt') as f:
        _ = f.readline()  # Skip the header line
        data_chunk = []
        data_size = 0

        # Read file line-by-line and accumulate until max size is reached
        for line in f:
            data_chunk.append(line)
            data_size += len(line.encode('utf-8'))  # Track size in bytes

            if data_size >= max_chunk_size:
                # Write the chunk in a separate process for better performance
                chunk_data = ''.join(data_chunk)
                multiprocessing.Process(target=write_chunk, args=(chunk_data, chunk_num)).start()
                data_chunk = []
                data_size = 0
                chunk_num += 1

        # Write any remaining lines as the last chunk
        if data_chunk:
            chunk_data = ''.join(data_chunk)
            multiprocessing.Process(target=write_chunk, args=(chunk_data, chunk_num)).start()

    print(f"Chunks with headers created in directory: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split a large CSV file into smaller chunks by file size")
    parser.add_argument("input_file", type=str, help="Path to the input CSV file (in .csv.gz format)")
    parser.add_argument("--max_chunk_size", type=int, default=50 * 1024 * 1024, help="Maximum chunk size in bytes (default 50 MB)")

    args = parser.parse_args()

    split_csv_by_size(args.input_file, max_chunk_size=args.max_chunk_size)
