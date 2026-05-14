import json
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset

BASE_DIR   = Path("/home/fireblue/datasets/eval/pope")
IMAGES_DIR = BASE_DIR / "images"
OUTPUT     = BASE_DIR / "test.jsonl"

BASE_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("🚀 下載 POPE Full (lmms-lab/POPE)...")
    raw = load_dataset("lmms-lab/POPE", "Full")
    split_name = list(raw.keys())[0]
    ds = raw[split_name]
    print(f"  → {len(ds)} 筆，split: {split_name}，欄位: {ds.column_names}")

    # 如果有 category 欄位，只取 adversarial（最具鑑別力）
    category_field = next((f for f in ["category", "type", "pope_type"] if f in ds.column_names), None)
    if category_field:
        ds = ds.filter(lambda x: x[category_field] == "adversarial")
        print(f"  → 篩選 adversarial 後: {len(ds)} 筆")
    else:
        print("  → 無 category 欄位，使用全部資料")

    with open(OUTPUT, "w", encoding="utf-8") as f:
        for i, ex in enumerate(tqdm(ds)):
            img_path = IMAGES_DIR / f"{i:06d}.jpg"
            if not img_path.exists():
                ex["image"].convert("RGB").save(img_path, format="JPEG")

            rec = {
                "question_id": ex.get("question_id", i),
                "question":    ex["question"],
                "answer":      ex["answer"].lower(),   # "yes" / "no"
                "image":       str(img_path),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"✅ {sum(1 for _ in open(OUTPUT))} 筆 → {OUTPUT}")

if __name__ == "__main__":
    main()
