import os
import pandas as pd
import base64
from PIL import Image
import io
import ast

def verify_images():
    base_dir = os.path.join(os.getcwd(), 'MPCC_HF')
    if not os.path.exists(base_dir):
        base_dir = os.path.join(os.getcwd(), 'MPCC')
    
    output_dir = "verified_images_sample"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Checking datasets in {base_dir}...")
    
    parquet_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.parquet'):
                parquet_files.append(os.path.join(root, file))
    
    parquet_files.sort()
    
    for file_path in parquet_files:
        filename = os.path.basename(file_path)
        print(f"\nVerifying {filename}...")
        
        try:
            df = pd.read_parquet(file_path)
            if df.empty:
                print("  Dataset is empty.")
                continue
                
            # Get the first row
            row = df.iloc[0]
            
            if 'image' not in row:
                print("  No 'image' column found.")
                continue
                
            image_data = row['image']
            
            # Handle list of images (it seems to be a string representation of a list based on previous output)
            # or a list object
            images_list = []
            if isinstance(image_data, str):
                try:
                    # Try to parse string as list
                    images_list = ast.literal_eval(image_data)
                except:
                    # Maybe it's just a single base64 string?
                    images_list = [image_data]
            elif isinstance(image_data, list) or isinstance(image_data, np.ndarray):
                images_list = image_data
            else:
                print(f"  Unknown image data type: {type(image_data)}")
                continue
                
            print(f"  Found {len(images_list)} images in the first sample.")
            
            # Verify the first image in the list
            if len(images_list) > 0:
                img_str = images_list[0]
                
                # Check if it looks like base64
                # Sometimes it might have a prefix like "data:image/jpeg;base64," which needs removal
                if img_str.startswith('data:image'):
                    img_str = img_str.split(',')[1]
                
                try:
                    img_bytes = base64.b64decode(img_str)
                    img = Image.open(io.BytesIO(img_bytes))
                    img.load() # Force load image data
                    
                    print(f"  ✅ SUCCESS: Image loaded successfully. Format: {img.format}, Size: {img.size}, Mode: {img.mode}")
                    
                    # Save sample
                    save_name = f"{os.path.splitext(filename)[0]}_sample.jpg"
                    save_path = os.path.join(output_dir, save_name)
                    img.save(save_path)
                    print(f"  Saved sample to: {save_path}")
                    
                except Exception as e:
                    print(f"  ❌ FAILED to decode/load image: {e}")
            else:
                print("  Image list is empty.")
                
        except Exception as e:
            print(f"  Error processing file: {e}")

if __name__ == "__main__":
    verify_images()
