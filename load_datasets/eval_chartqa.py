import json
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset

# 對齊 default.yaml: chartqa.data_path / image_dir
BASE_DIR   = Path("/home/fireblue/datasets/eval/chartqa")
IMAGES_DIR = BASE_DIR / "images"
OUTPUT     = BASE_DIR / "test.jsonl"

BASE_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("🚀 下載 ChartQA test split (lmms-lab/ChartQA)...")
    ds = load_dataset("lmms-lab/ChartQA", split="test")

    with open(OUTPUT, "w", encoding="utf-8") as f:
        for i, ex in enumerate(tqdm(ds)):
            img_path = IMAGES_DIR / f"{i:06d}.png"
            if not img_path.exists():
                ex["image"].convert("RGB").save(img_path, format="PNG")

            # eval script: inst["image"].split("/")[-1] → join with image_dir
            rec = {
                "type":     ex["type"],
                "question": ex["question"],
                "answer":   ex["answer"],
                "image":    str(img_path),   # 絕對路徑，split("/")[-1] 取檔名
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"✅ {sum(1 for _ in open(OUTPUT))} 筆 → {OUTPUT}")

if __name__ == "__main__":
    main()
