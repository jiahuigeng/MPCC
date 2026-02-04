import os
import pandas as pd

def inspect_datasets():
    # Define the base directory relative to the current working directory
    # Based on file structure, data is in ./MPCC_HF/ (newly downloaded) or ./MPCC/
    base_dir = os.path.join(os.getcwd(), 'MPCC_HF')
    if not os.path.exists(base_dir):
        base_dir = os.path.join(os.getcwd(), 'MPCC')
    
    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} does not exist.")
        return

    # Walk through the directory to find parquet files
    parquet_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.parquet'):
                parquet_files.append(os.path.join(root, file))
    
    # Sort files to have a consistent order
    parquet_files.sort()

    if not parquet_files:
        print("No .parquet files found.")
        return

    print(f"Found {len(parquet_files)} parquet files. Displaying first 2 rows for each:\n")

    for file_path in parquet_files:
        print("="*80)
        print(f"File: {os.path.basename(file_path)}")
        print(f"Path: {file_path}")
        print("-" * 80)
        
        try:
            # Read the parquet file
            df = pd.read_parquet(file_path)
            
            # Display first 2 rows
            # Using to_string() to ensure full content is visible if possible, 
            # or just default repr which is usually good for pandas
            print(df.head(2))
            print("\n")
        except Exception as e:
            print(f"Error reading file: {e}\n")

if __name__ == "__main__":
    inspect_datasets()
