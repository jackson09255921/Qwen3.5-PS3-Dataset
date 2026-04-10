import os
import shutil
import subprocess
from pathlib import Path

BASE_DIR = Path("./")
REMOTE_DIR = "gdrive:datasets"

datasets = [
    "CodeAlpaca",
    "DataOptim",
    "Idatabricks",
    "SVIT",
    "llava_instruct",
    "math500",
    "math_qa",
    "megachat_gpt4o",
    "tulu_personas",
]

for name in datasets:
    folder = BASE_DIR / name
    if not folder.exists():
        print(f"[skip] {folder} 不存在")
        continue

    zip_base = BASE_DIR / name            # 不含 .zip 副檔名
    zip_path = BASE_DIR / f"{name}.zip"   # 實際 zip 檔案

    # 1. 壓縮資料夾
    print(f"[zip] {folder} -> {zip_path}")
    shutil.make_archive(
        base_name=str(zip_base),   # /home/.../CodeAlpaca
        format="zip",
        root_dir=str(folder),      # 要壓的資料夾
    )

    # 2. 上傳 zip 到 Google Drive
    print(f"[upload] {zip_path} -> {REMOTE_DIR}")
    cmd = [
        "rclone",
        "copy",
        str(zip_path),
        REMOTE_DIR,
        "--progress",
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[error] 上傳失敗: {zip_path}，保留 zip 以便重試")
        break

    # 3. 上傳成功後刪除本地 zip
    print(f"[clean] delete local {zip_path}")
    os.remove(zip_path)
