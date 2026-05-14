import json
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset

BASE_DIR   = Path("/home/fireblue/datasets/eval/vsr")
IMAGES_DIR = BASE_DIR / "images"
OUTPUT     = BASE_DIR / "test.jsonl"

BASE_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("🚀 下載 VSR test split (cambridgeltl/visual_spatial_reasoning)...")
    ds = load_dataset("cambridgeltl/visual_spatial_reasoning", split="test")

    with open(OUTPUT, "w", encoding="utf-8") as f:
        for i, ex in enumerate(tqdm(ds)):
            img_path = IMAGES_DIR / f"{i:06d}.jpg"
            if not img_path.exists():
                ex["image"].convert("RGB").save(img_path, format="JPEG")

            rec = {
                "question_id": i,
                "caption":     ex["caption"],
                "answer":      "true" if ex["label"] else "false",
                "relation":    ex.get("relation", ""),
                "image":       str(img_path),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"✅ {sum(1 for _ in open(OUTPUT))} 筆 → {OUTPUT}")

if __name__ == "__main__":
    main()
