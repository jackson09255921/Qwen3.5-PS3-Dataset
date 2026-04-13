import json
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset

# 對齊 default.yaml: infographicvqa.data_path / image_dir
# 圖片與 DocVQA 共用同一個 images/ 資料夾，用 infovqa_ 前綴避免衝突
BASE_DIR   = Path("/home/fireblue/datasets/eval/docvqa")
IMAGES_DIR = BASE_DIR / "images"
OUTPUT     = BASE_DIR / "infographicvqa.jsonl"   # eval script: raw["data"]

BASE_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("🚀 下載 InfographicVQA validation split (lmms-lab/DocVQA, InfographicVQA)...")
    # validation split 有 answers，test split 的 answers = None
    ds = load_dataset("lmms-lab/DocVQA", "InfographicVQA", split="validation")

    records = []
    for i, ex in enumerate(tqdm(ds)):
        img_name = f"infovqa_{i:06d}.png"
        img_path = IMAGES_DIR / img_name
        if not img_path.exists():
            ex["image"].convert("RGB").save(img_path, format="PNG")

        answers = ex.get("answers") or []
        if isinstance(answers, str):
            answers = [answers]

        records.append({
            "questionId": ex.get("questionId", i),
            "question":   ex["question"],
            "answers":    answers,
            "image":      img_name,   # eval script: os.path.join(image_dir, img_name)
        })

    # eval script (docvqa.py): raw = io.load(data_path) → raw["data"]
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump({"data": records}, f, ensure_ascii=False, indent=2)

    print(f"✅ {len(records)} 筆 → {OUTPUT}")
    print(f"   圖片：{IMAGES_DIR}")

if __name__ == "__main__":
    main()
