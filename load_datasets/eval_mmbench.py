import json
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset

BASE_DIR   = Path("/home/fireblue/datasets/eval/mmbench")
IMAGES_DIR = BASE_DIR / "images"
OUTPUT     = BASE_DIR / "test.jsonl"

BASE_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

LETTERS = ["A", "B", "C", "D"]


def main():
    print("下載 HuggingFaceM4/MMBench_dev ...")
    ds = load_dataset("HuggingFaceM4/MMBench_dev", split="train")
    print(f"  → {len(ds)} 筆，欄位: {ds.column_names}")

    records = []
    for i, ex in enumerate(tqdm(ds)):
        img_name = f"{i:06d}.jpg"
        img_path = IMAGES_DIR / img_name
        if not img_path.exists():
            ex["image"].convert("RGB").save(img_path, format="JPEG")

        # label 是 int 0-3
        label = ex.get("label", 0)
        answer = LETTERS[label] if isinstance(label, int) and label < 4 else str(label).upper()

        # 組選項（A/B/C/D 欄位）
        options = [f"{l}. {ex[l]}" for l in LETTERS if ex.get(l)]

        question = ex.get("question", "")
        hint = ex.get("hint", "") or ""
        if hint:
            question = f"{hint}\n{question}"

        records.append({
            "question_id": ex.get("index", i),
            "question":    question,
            "options":     options,
            "answer":      answer,
            "category":    ex.get("category", ""),
            "image":       str(img_path),
        })

    with open(OUTPUT, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n✅ {len(records)} 筆 → {OUTPUT}")


if __name__ == "__main__":
    main()
