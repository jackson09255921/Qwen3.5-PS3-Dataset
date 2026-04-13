import json
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset

# 對齊 default.yaml: ocrbench.data_path / image_dir
BASE_DIR   = Path("/home/fireblue/datasets/eval/ocrbench")
IMAGES_DIR = BASE_DIR / "images"
OUTPUT     = BASE_DIR / "ocrbench.jsonl"

BASE_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("🚀 下載 OCRBench test split (echo840/OCRBench)...")
    ds = load_dataset("echo840/OCRBench", split="test")

    with open(OUTPUT, "w", encoding="utf-8") as f:
        for i, ex in enumerate(tqdm(ds)):
            img_name = f"{i:06d}.jpg"
            img_path = IMAGES_DIR / img_name
            if not img_path.exists():
                ex["image"].convert("RGB").save(img_path, format="JPEG")

            rec = {
                "question_id":   i,
                "dataset":       ex["dataset"],
                "question":      ex["question"],
                "question_type": ex["question_type"],
                "answers":       ex["answer"] if isinstance(ex["answer"], list) else [ex["answer"]],
                "image":         img_name,   # relative filename, joined with image_dir at eval time
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"✅ {sum(1 for _ in open(OUTPUT))} 筆 → {OUTPUT}")

if __name__ == "__main__":
    main()
