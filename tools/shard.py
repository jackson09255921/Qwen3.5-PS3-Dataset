from __future__ import annotations
import os
from pathlib import Path
import math
import subprocess

DATA_DIR = Path("coyo-700m-3M")
IMAGES_DIR = DATA_DIR / "images"
SHARDS_DIR = DATA_DIR / "shards"

# 每個 shard 目標張數，依你實際平均大小調一下
IMAGES_PER_SHARD = 120_000


def main():
    SHARDS_DIR.mkdir(exist_ok=True)
    files = sorted([p for p in IMAGES_DIR.iterdir() if p.is_file()])
    n = len(files)
    num_shards = math.ceil(n / IMAGES_PER_SHARD)
    print(f"total images = {n}, shards = {num_shards}")

    for i in range(num_shards):
        start = i * IMAGES_PER_SHARD
        end = min((i + 1) * IMAGES_PER_SHARD, n)
        shard_files = files[start:end]
        list_path = SHARDS_DIR / f"shard_{i:04d}.txt"
        tar_path = SHARDS_DIR / f"shard_{i:04d}.tar"

        if tar_path.exists():
            print(f"[skip] {tar_path} already exists")
            continue

        # 建立檔案清單（相對於 DATA_DIR）
        with list_path.open("w") as f:
            for p in shard_files:
                rel = p.relative_to(DATA_DIR)
                f.write(str(rel) + "\n")

        print(f"creating {tar_path} for images {start}..{end-1}")
        # 從 DATA_DIR 下打包，確保 tar 裡路徑是 images/xxx.jpg
        subprocess.run(
            ["tar", "-cf", str(tar_path), "-C", str(DATA_DIR), "-T", str(list_path)],
            check=True,
        )

    print("done. check shard sizes with: du -sh coyo-700m-3M/shards/*.tar")


if __name__ == "__main__":
    main()
