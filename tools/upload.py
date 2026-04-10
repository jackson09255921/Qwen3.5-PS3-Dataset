import os
import json
import math
import shutil
import tarfile
from pathlib import Path
from typing import List

from huggingface_hub import HfApi

USER = "FireBlueOnly"

# ====== 你可以調的參數 ======
SOURCE_DIR = "../coyo-700m-3M"          # 原始資料夾
SOURCE_CHAT = "chat.jsonl"             # 原始 chat 檔名
SOURCE_IMAGES_DIR = "images"           # 原始圖片資料夾

TARGET_REPO_SIZE_GB = 300              # 每個 repo 目標容量 (GiB 粗估)
MAX_TAR_SIZE_GB = 40                   # 單一 tar 最大容量 (GiB，< 50GB)
BASE_REPO_NAME = "coyo-700M-3M-PS3"    # HF 上的 base 名稱
PRIVATE = False                        # True=private, False=public
# ===========================


def get_total_dataset_size(source_dir: Path) -> int:
    chat_path = source_dir / SOURCE_CHAT
    images_dir = source_dir / SOURCE_IMAGES_DIR

    total = 0
    if chat_path.is_file():
        total += chat_path.stat().st_size
    for p in images_dir.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def decide_num_subsets(total_bytes: int) -> int:
    target_repo_bytes = int(TARGET_REPO_SIZE_GB * (1024 ** 3))
    num = max(1, math.ceil(total_bytes / target_repo_bytes))
    print(f"[auto] total size ~ {total_bytes / (1024**3):.2f} GiB, "
          f"target {TARGET_REPO_SIZE_GB} GiB per repo -> NUM_SUBSETS = {num}")
    return num


def upload_dataset_files(tar_root: str, repo_name: str, private: bool):
    """
    一個一個檔案 (tar + README) 用 upload_file 上傳，避免 upload_large_folder 的不穩。
    """
    repo_id = f"{USER}/{repo_name}"
    api = HfApi()

    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True,
    )
    print(f"repo ready: {repo_id}")

    tar_root_path = Path(tar_root)
    files_to_upload: List[Path] = []

    for p in tar_root_path.iterdir():
        if not p.is_file():
            continue
        if p.suffix == ".tar" or p.name == "README.md":
            files_to_upload.append(p)

    files_to_upload = sorted(files_to_upload, key=lambda x: x.name)

    for p in files_to_upload:
        print(f"Uploading {p.name} ...")
        api.upload_file(
            path_or_fileobj=str(p),
            path_in_repo=p.name,
            repo_id=repo_id,
            repo_type="dataset",
        )

    print(f"uploaded: {repo_id}")


def write_readme_with_yaml(tar_root: Path, subset_tag: str, tar_names: List[str]):
    readme_path = tar_root / "README.md"
    print(f"Writing README with YAML to {readme_path}")

    data_files_yaml = ""
    for name in tar_names:
        data_files_yaml += f"      - split: train\n        path: \"{name}\"\n"

    yaml_header = f"""---
configs:
  - config_name: default
    data_files:
{data_files_yaml}---

# {BASE_REPO_NAME} Subset {subset_tag}

This subset contains a portion of COYO-700M-3M preprocessed for PS3.
"""
    readme_path.write_text(yaml_header, encoding="utf-8")


def pack_subset_lines_to_tars(
    subset_tag: str,
    subset_lines: List[str],
    source_dir: Path,
    tar_root: Path,
) -> List[Path]:
    """
    直接根據 subset_lines 從原始 images 打多個 tar，不建立 subset/images。
    """
    max_bytes = int(MAX_TAR_SIZE_GB * (1024 ** 3))
    images_dir = source_dir / SOURCE_IMAGES_DIR

    base_name = f"{source_dir.name}-{subset_tag}"
    tars: List[Path] = []
    tar_idx = 1

    current_tar = None
    current_tar_path = None
    current_size = 0
    current_chat_lines: List[str] = []
    seen_images_in_tar = set()

    def start_new_tar():
        nonlocal current_tar, current_tar_path, current_size, tar_idx, current_chat_lines, seen_images_in_tar
        if current_tar is not None:
            tmp_chat = tar_root / f"{base_name}-{tar_idx-1:05d}.chat.jsonl"
            with tmp_chat.open("w", encoding="utf-8") as cf:
                for l in current_chat_lines:
                    cf.write(l)
            current_tar.add(tmp_chat, arcname="chat.jsonl")
            current_size += tmp_chat.stat().st_size
            tmp_chat.unlink()
            current_tar.close()

        tar_name = f"{base_name}-{tar_idx:05d}.tar"
        current_tar_path = tar_root / tar_name
        print(f"Starting new tar: {current_tar_path}")
        current_tar = tarfile.open(current_tar_path, "w")
        current_size = 0
        current_chat_lines = []
        seen_images_in_tar = set()
        tars.append(current_tar_path)
        tar_idx += 1

    start_new_tar()

    for line in subset_lines:
        if not line.strip():
            continue
        data = json.loads(line)

        image_field = data.get("image")
        if image_field is None:
            continue

        img_name = os.path.basename(image_field)
        src_img = images_dir / img_name
        if not src_img.is_file():
            print(f"Warning: image not found: {src_img}")
            continue

        img_size = src_img.stat().st_size
        approx_sample_size = img_size + 1024

        if current_size + approx_sample_size > max_bytes and current_size > 0:
            start_new_tar()

        data["image"] = f"images/{img_name}"
        current_chat_lines.append(json.dumps(data, ensure_ascii=False) + "\n")

        if img_name not in seen_images_in_tar:
            current_tar.add(src_img, arcname=f"images/{img_name}")
            current_size += img_size
            seen_images_in_tar.add(img_name)

    if current_tar is not None and current_chat_lines:
        tmp_chat = tar_root / f"{base_name}-{tar_idx-1:05d}.chat.jsonl"
        with tmp_chat.open("w", encoding="utf-8") as cf:
            for l in current_chat_lines:
                cf.write(l)
        current_tar.add(tmp_chat, arcname="chat.jsonl")
        current_size += tmp_chat.stat().st_size
        tmp_chat.unlink()
        current_tar.close()

    return tars


def split_and_upload_one_by_one():
    source_dir = Path(SOURCE_DIR)
    chat_path = source_dir / SOURCE_CHAT
    images_dir = source_dir / SOURCE_IMAGES_DIR

    assert chat_path.is_file(), f"{chat_path} 不存在"
    assert images_dir.is_dir(), f"{images_dir} 不存在"

    total_bytes = get_total_dataset_size(source_dir)
    num_subsets = decide_num_subsets(total_bytes)

    print(f"Loading chat from {chat_path} ...")
    with chat_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    total = len(lines)
    print(f"Total samples: {total}")
    assert total > 0, "chat.jsonl 是空的？"

    subset_size = math.ceil(total / num_subsets)
    print(f"Splitting into {num_subsets} subsets, ~{subset_size} samples each")

    for subset_idx in range(num_subsets):
        start = subset_idx * subset_size
        end = min((subset_idx + 1) * subset_size, total)
        if start >= end:
            break

        subset_lines = lines[start:end]
        human_idx = subset_idx + 1
        subset_tag = f"{human_idx:02d}"

        print(f"\n=== [subset {subset_tag}] samples: {len(subset_lines)} ===")

        tar_root = source_dir.parent / f"{source_dir.name}-tars-{subset_tag}"
        tar_root.mkdir(parents=True, exist_ok=True)

        tar_paths = pack_subset_lines_to_tars(
            subset_tag=subset_tag,
            subset_lines=subset_lines,
            source_dir=source_dir,
            tar_root=tar_root,
        )
        tar_names = [p.name for p in tar_paths]

        write_readme_with_yaml(tar_root, subset_tag, tar_names)

        repo_name = f"{BASE_REPO_NAME}-{subset_tag}"
        print(f"\n=== Uploading tars for subset {subset_tag} to {USER}/{repo_name} ===")

        upload_dataset_files(
            tar_root=str(tar_root),
            repo_name=repo_name,
            private=PRIVATE,
        )

        print(f"Removing local tar_root dir: {tar_root}")
        shutil.rmtree(tar_root)


def main():
    split_and_upload_one_by_one()


if __name__ == "__main__":
    main()
