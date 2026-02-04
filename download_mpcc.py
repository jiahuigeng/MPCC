from huggingface_hub import snapshot_download
import os

repo_id = "jyyyyy67/MPCC"
local_dir = "MPCC_HF"

print(f"Downloading dataset from {repo_id} to {local_dir}...")
try:
    snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=local_dir, local_dir_use_symlinks=False)
    print("Download complete.")
except Exception as e:
    print(f"Failed to download: {e}")
