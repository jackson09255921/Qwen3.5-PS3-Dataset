import json
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset

# 對齊 default.yaml: textvqa.val_data_path / val_answer_path / image_dir
BASE_DIR       = Path("/home/fireblue/datasets/eval/textvqa")
IMAGES_DIR     = BASE_DIR / "images"
QUESTIONS_FILE = BASE_DIR / "val.jsonl"    # eval script: instance["image"], instance["text"], instance["question_id"]
ANSWERS_FILE   = BASE_DIR / "answer.jsonl" # eval script: answers["data"] → {(image_id, question): annotation}

BASE_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("🚀 下載 TextVQA validation split (lmms-lab/textvqa)...")
    ds = load_dataset("lmms-lab/textvqa", split="validation")

    answers_data = []

    with open(QUESTIONS_FILE, "w", encoding="utf-8") as fq:
        for i, ex in enumerate(tqdm(ds)):
            img_name = f"{i}.jpg"
            img_path = IMAGES_DIR / img_name
            if not img_path.exists():
                ex["image"].convert("RGB").save(img_path, format="JPEG")

            # eval script lookup key: (result["question_id"], processed_prompt)
            # answers dict key:       (annotation["image_id"], question.lower())
            # → question_id 必須等於 image_id，用 sequential index i 統一兩者
            fq.write(json.dumps({
                "question_id": i,
                "image_id":    i,
                "image":       img_name,
                "text":        ex["question"],   # eval script: instance["text"]
            }, ensure_ascii=False) + "\n")

            answers_data.append({
                "image_id": i,               # 對應 question_id
                "question": ex["question"],
                "answers":  ex["answers"],
            })

    # eval script: io.load(answer_path)["data"]
    with open(ANSWERS_FILE, "w", encoding="utf-8") as fa:
        json.dump({"data": answers_data}, fa, ensure_ascii=False, indent=2)

    print(f"✅ {len(answers_data)} 筆")
    print(f"   問題：{QUESTIONS_FILE}")
    print(f"   答案：{ANSWERS_FILE}")

if __name__ == "__main__":
    main()
