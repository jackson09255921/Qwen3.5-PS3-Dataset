import json
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset

# 對齊 default.yaml: docvqa.data_path / image_dir
BASE_DIR   = Path("/home/fireblue/datasets/eval/docvqa")
IMAGES_DIR = BASE_DIR / "images"
OUTPUT     = BASE_DIR / "docvqa.jsonl"   # eval script: raw["data"]

BASE_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("🚀 下載 DocVQA val split (vikhyatk/docvqa-val)...")
    # vikhyatk/docvqa-val 有完整 answers，lmms-lab/DocVQA val/test 的 answers 為 None
    ds = load_dataset("vikhyatk/docvqa-val", split="validation")

    records = []
    qid = 0
    for idx, sample in enumerate(tqdm(ds)):
        img_name = f"{idx}.png"
        img_path = IMAGES_DIR / img_name
        if not img_path.exists():
            sample["image"].convert("RGB").save(img_path, format="PNG")

        # 一張圖有多個 QA pair，各自建一筆記錄
        for qa in sample["qa"]:
            records.append({
                "questionId": qid,
                "question":   qa["question"],
                "answers":    qa["answers"],
                "image":      img_name,    # eval script: os.path.join(image_dir, img_name)
            })
            qid += 1

    # eval script: raw = io.load(data_path) → raw["data"]
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump({"data": records}, f, ensure_ascii=False, indent=2)

    print(f"✅ {len(records)} 筆（{idx+1} 張圖）→ {OUTPUT}")

if __name__ == "__main__":
    main()
