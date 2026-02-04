import os
import pandas as pd

def print_dataset_keys():
    # 优先使用新下载的 MPCC_HF 目录，如果不存在则尝试 MPCC
    base_dir = os.path.join(os.getcwd(), 'MPCC_HF')
    if not os.path.exists(base_dir):
        base_dir = os.path.join(os.getcwd(), 'MPCC')
    
    print(f"Checking datasets in {base_dir}...\n")
    
    # 获取所有 .parquet 文件
    parquet_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.parquet'):
                parquet_files.append(os.path.join(root, file))
    
    parquet_files.sort()
    
    # 存储不同文件结构的 key 集合，避免重复打印完全相同的结构
    seen_columns = set()

    for file_path in parquet_files:
        filename = os.path.basename(file_path)
        try:
            # 只读取列名，不读取数据，提高速度
            df = pd.read_parquet(file_path)
            columns = tuple(df.columns.tolist())
            
            # 如果是新的列结构，或者是第一个文件，打印出来
            # 也可以选择打印每个文件的 keys，这里为了清晰打印每个文件的 keys
            print(f"File: {filename}")
            print(f"Keys: {list(columns)}")
            print("-" * 50)
            
        except Exception as e:
            print(f"Error reading {filename}: {e}")

if __name__ == "__main__":
    print_dataset_keys()
