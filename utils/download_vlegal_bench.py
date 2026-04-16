import os
from huggingface_hub import snapshot_download

def download_dataset():
    repo_id = "CMC-OPENAI/VLegal-Bench"
    local_dir = "data/vlegal_bench"
    
    print(f"[INFO] Starting download of dataset: {repo_id}")
    print(f"[INFO] Destination: {os.path.abspath(local_dir)}")
    
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )
        print(f"\n[SUCCESS] Dataset downloaded successfully to {local_dir}")
        
        # List files to verify
        print("[INFO] Downloaded files:")
        for root, dirs, files in os.walk(local_dir):
            for file in files:
                print(f"  - {os.path.join(root, file)}")
                
    except Exception as e:
        print(f"\n[ERROR] Failed to download dataset: {e}")

if __name__ == "__main__":
    download_dataset()
