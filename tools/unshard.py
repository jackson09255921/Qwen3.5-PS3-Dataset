from __future__ import annotations
import os
from pathlib import Path
import subprocess

DATA_DIR = Path("coyo-700m-3M")
IMAGES_DIR = DATA_DIR / "images"
SHARDS_DIR = DATA_DIR / "shards"

def main():
    IMAGES_DIR.mkdir(exist_ok=True)
    
    # 找到所有 shard tar 檔案
    tar_files = sorted([p for p in SHARDS_DIR.iterdir() if p.suffix == '.tar'])
    print(f"Found {len(tar_files)} shard tar files")
    
    for tar_path in tar_files:
        print(f"Extracting {tar_path.name}...")
        
        # 使用 tar 解壓到 DATA_DIR，會自動還原 images/ 結構
        subprocess.run(
            ["tar", "-xf", str(tar_path), "-C", str(DATA_DIR)],
            check=True,
        )
        print(f"✓ Extracted {tar_path.name}")
    
    # 檢查總圖片數
    total_images = len(list(IMAGES_DIR.iterdir()))
    print(f"All done! Total images in {IMAGES_DIR}: {total_images:,}")
    print("Verify with: find images -type f | wc -l")

if __name__ == "__main__":
    main()
