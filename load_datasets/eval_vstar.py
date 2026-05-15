import json
import os
from io import BytesIO
from pathlib import Path

from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import snapshot_download

BASE_DIR   = Path("/home/fireblue/datasets/eval/vstar")
IMAGES_DIR = BASE_DIR / "images"
OUTPUT     = BASE_DIR / "test.jsonl"

BASE_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

LETTERS = ["A", "B", "C", "D"]


def _get_split(raw):
    for prefer in ["test", "validation", "train"]:
        if prefer in raw:
            return raw[prefer]
    return raw[list(raw.keys())[0]]


def _load_image(img_ref, dataset_root: str = "") -> Image.Image:
    """Handle PIL Image / dict{"bytes","path"} / relative string path."""
    if hasattr(img_ref, "convert"):
        return img_ref.convert("RGB")
    if isinstance(img_ref, dict):
        if img_ref.get("bytes"):
            return Image.open(BytesIO(img_ref["bytes"])).convert("RGB")
        if img_ref.get("path"):
            return Image.open(img_ref["path"]).convert("RGB")
    if isinstance(img_ref, str):
        # 嘗試相對於 dataset_root 拼完整路徑
        full = os.path.join(dataset_root, img_ref) if dataset_root else img_ref
        if os.path.exists(full):
            return Image.open(full).convert("RGB")
        if os.path.exists(img_ref):
            return Image.open(img_ref).convert("RGB")
        raise FileNotFoundError(f"Image not found: {full}")
    raise ValueError(f"Cannot load image from type {type(img_ref)}")


def _answer_letter(ex) -> str:
    raw = ex.get("label", ex.get("answer", ex.get("answer_idx", 0)))
    if isinstance(raw, int):
        return LETTERS[raw] if raw < len(LETTERS) else str(raw)
    raw = str(raw).strip()
    if raw.upper() in LETTERS:
        return raw.upper()
    return raw


def main():
    print("下載 craigwu/vstar_bench ...")
    # snapshot_download 拿到 dataset 檔案在本地的根目錄（含圖片相對路徑）
    dataset_root = snapshot_download(repo_id="craigwu/vstar_bench", repo_type="dataset")
    print(f"  dataset_root: {dataset_root}")
    ds = _get_split(load_dataset("craigwu/vstar_bench"))
    print(f"  → {len(ds)} 筆，欄位: {ds.column_names}")

    # 找 subset / category 欄位
    subset_field = next(
        (f for f in ["subset", "category", "type"] if f in ds.column_names), None
    )
    # 找問題欄位
    question_field = next(
        (f for f in ["text", "question", "query"] if f in ds.column_names), None
    )

    records = []
    for i, ex in enumerate(tqdm(ds)):
        img_name = f"{i:06d}.jpg"
        img_path = IMAGES_DIR / img_name
        if not img_path.exists():
            _load_image(ex["image"], dataset_root=dataset_root).save(img_path, format="JPEG")

        subset = ex[subset_field] if subset_field else "default"
        question = ex[question_field] if question_field else ""
        answer = _answer_letter(ex)

        records.append({
            "question_id": ex.get("question_id", i),
            "subset":      subset,
            "question":    question,   # text 欄位已含選項
            "answer":      answer,
            "image":       str(img_path),
        })

    with open(OUTPUT, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 統計
    subset_counts: dict = {}
    for r in records:
        subset_counts[r["subset"]] = subset_counts.get(r["subset"], 0) + 1

    print(f"\n✅ {len(records)} 筆 → {OUTPUT}")
    for s, n in subset_counts.items():
        print(f"   {s}: {n}")


if __name__ == "__main__":
    main()
