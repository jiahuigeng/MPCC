import os
import argparse
import ast
import base64
import io
import pandas as pd
from PIL import Image

def parse_list(value):
    if isinstance(value, str):
        s = value.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                return ast.literal_eval(s)
            except Exception:
                return [value]
        return [value]
    if isinstance(value, (list, tuple)):
        return list(value)
    return []

def to_bytes(item):
    if isinstance(item, (bytes, bytearray, memoryview)):
        return bytes(item)
    if isinstance(item, str):
        s = item
        if s.startswith("data:image"):
            parts = s.split(",", 1)
            s = parts[1] if len(parts) > 1 else s
        try:
            return base64.b64decode(s)
        except Exception:
            return None
    return None

def save_image(img_bytes, path):
    try:
        img = Image.open(io.BytesIO(img_bytes))
        img.save(path, format="JPEG")
        return True
    except Exception:
        return False

def extract_from_parquet(parquet_path, out_dir, limit=None):
    df = pd.read_parquet(parquet_path)
    group = os.path.basename(os.path.dirname(parquet_path))
    base_name = os.path.splitext(os.path.basename(parquet_path))[0]
    target_dir = os.path.join(out_dir, group, base_name)
    os.makedirs(target_dir, exist_ok=True)

    if limit is not None:
        df = df.head(limit)

    for _, row in df.iterrows():
        idx = row.get("index")
        images = parse_list(row.get("image"))
        names = parse_list(row.get("image_path"))
        for j, item in enumerate(images):
            b = to_bytes(item)
            if b is None:
                continue
            if names and j < len(names) and isinstance(names[j], str) and names[j]:
                fname = names[j]
            else:
                fname = f"{base_name}_row{idx}_img{j+1}.jpg"
            save_path = os.path.join(target_dir, fname)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            ok = save_image(b, save_path)
            if not ok:
                alt_name = f"{base_name}_row{idx}_img{j+1}.jpg"
                save_path = os.path.join(target_dir, alt_name)
                save_image(b, save_path)

def find_parquets(base_dir):
    files = []
    for root, _, fs in os.walk(base_dir):
        for f in fs:
            if f.endswith(".parquet"):
                files.append(os.path.join(root, f))
    files.sort()
    return files

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=str, default=None)
    parser.add_argument("--base", type=str, default="MPCC_HF")
    parser.add_argument("--out", type=str, default="extracted_images")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    base_dir = os.path.join(os.getcwd(), args.base)
    if not os.path.exists(base_dir):
        alt = os.path.join(os.getcwd(), "MPCC")
        base_dir = alt if os.path.exists(alt) else base_dir

    os.makedirs(args.out, exist_ok=True)

    if args.parquet:
        extract_from_parquet(args.parquet, args.out, args.limit)
    else:
        for p in find_parquets(base_dir):
            extract_from_parquet(p, args.out, args.limit)

if __name__ == "__main__":
    main()
