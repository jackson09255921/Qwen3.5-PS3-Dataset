import json
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset

BASE_DIR   = Path("/home/fireblue/datasets/eval/gqa")
IMAGES_DIR = BASE_DIR / "images"
OUTPUT     = BASE_DIR / "testdev_balanced.jsonl"

BASE_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

def _field(ex, *candidates):
    for c in candidates:
        if c in ex:
            return ex[c]
    raise KeyError(f"None of {candidates} found in {list(ex.keys())}")

def _split(raw):
    return raw[list(raw.keys())[0]]

def main():
    # ── Step 1: save images, named by their actual image_id ──────────
    print("📥 下載 GQA images (testdev_balanced_images)...")
    ds_imgs = _split(load_dataset("lmms-lab/GQA", "testdev_balanced_images"))
    print(f"  → {len(ds_imgs)} 張，欄位: {ds_imgs.column_names}")

    for ex in tqdm(ds_imgs, desc="儲存圖片"):
        img_id  = str(_field(ex, "image_id", "imageId", "id"))
        img_path = IMAGES_DIR / f"{img_id}.jpg"
        if not img_path.exists():
            ex["image"].convert("RGB").save(img_path, format="JPEG")

    # ── Step 2: process instructions ─────────────────────────────────
    print("📥 下載 GQA instructions (testdev_balanced_instructions)...")
    ds_inst = _split(load_dataset("lmms-lab/GQA", "testdev_balanced_instructions"))
    print(f"  → {len(ds_inst)} 筆，欄位: {ds_inst.column_names}")

    with open(OUTPUT, "w", encoding="utf-8") as f:
        for ex in tqdm(ds_inst, desc="寫入 JSONL"):
            img_id  = str(_field(ex, "image_id", "imageId", "image_name"))
            qid     = str(_field(ex, "question_id", "questionId", "id"))
            rec = {
                "question_id": qid,
                "question":    ex["question"],
                "answer":      ex["answer"],
                "image":       str(IMAGES_DIR / f"{img_id}.jpg"),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"✅ {sum(1 for _ in open(OUTPUT))} 筆 → {OUTPUT}")

if __name__ == "__main__":
    main()
