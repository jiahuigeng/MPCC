import os
import pandas as pd
import argparse
from typing import Optional

def export_to_xlsx(limit: Optional[int] = None, output_file: str = "mpcc_data_summary.xlsx"):
    base_dir = os.path.join(os.getcwd(), 'MPCC_HF')
    if not os.path.exists(base_dir):
        base_dir = os.path.join(os.getcwd(), 'MPCC')
    
    print(f"Scanning datasets in {base_dir}...")
    
    parquet_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.parquet'):
                parquet_files.append(os.path.join(root, file))
    
    parquet_files.sort()
    
    # Create Excel Writer
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        for file_path in parquet_files:
            filename = os.path.basename(file_path)
            sheet_name = os.path.splitext(filename)[0][:31] # Excel sheet name limit is 31 chars
            
            print(f"Processing {filename} -> Sheet: {sheet_name}")
            
            try:
                df = pd.read_parquet(file_path)
                
                # Drop 'image' column to save space and make it readable
                if 'image' in df.columns:
                    df = df.drop(columns=['image'])
                
                # Truncate long text fields if necessary (optional, but helpful for Excel)
                # Just keeping it raw for now as requested
                
                if limit:
                    df = df.head(limit)
                
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                
    print(f"\n✅ Export complete! Data saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit rows per dataset (default: all)")
    parser.add_argument("--out", type=str, default="mpcc_data_summary.xlsx", help="Output Excel filename")
    args = parser.parse_args()
    
    export_to_xlsx(limit=args.limit, output_file=args.out)
