import json
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset

# ---------- 路徑（對齊 default.yaml）----------
BASE_DIR = Path("/home/fireblue/datasets/eval/textvqa")
IMAGES_DIR = BASE_DIR / "images"
VAL_QUESTIONS_FILE = BASE_DIR / "val.jsonl"     # yaml: val_data_path
VAL_ANSWERS_FILE   = BASE_DIR / "answer.jsonl"  # yaml: val_answer_path

BASE_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# ---------- 主程序 ----------
def main():
    print("🚀 下載 TextVQA validation split...")
    ds = load_dataset("lmms-lab/textvqa", split="validation")

    questions = []
    answers_data = []

    for item in tqdm(ds, desc="處理"):
        pil_img = item.get("image")
        question_id = item.get("question_id")
        image_id = item.get("image_id", question_id)
        question = str(item.get("question", "")).strip()
        gt_answers = item.get("answers", [])

        if not pil_img or question_id is None or not question:
            continue

        if isinstance(gt_answers, str):
            gt_answers = [gt_answers]

        img_name = f"{image_id}.jpg"
        save_path = IMAGES_DIR / img_name
        if not save_path.exists():
            try:
                pil_img.convert("RGB").save(save_path)
            except Exception:
                continue

        # 問題檔：eval script: instance["image"], instance["text"], instance["question_id"]
        questions.append({
            "question_id": question_id,
            "image_id": image_id,
            "image": img_name,
            "text": question,
        })

        # 答案檔：eval script: answers["data"] → {(image_id, question.lower()): annotation}
        answers_data.append({
            "question_id": question_id,
            "image_id": image_id,
            "question": question,
            "answers": gt_answers,
        })

    # 問題 JSONL（每行一筆，io.load 直接回傳 list）
    with open(VAL_QUESTIONS_FILE, "w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    # 答案 JSON（{"data": [...]}，io.load 後取 ["data"]）
    with open(VAL_ANSWERS_FILE, "w", encoding="utf-8") as f:
        json.dump({"data": answers_data}, f, ensure_ascii=False, indent=2)

    print(f"✅ 完成：{len(questions)} 筆")
    print(f"   問題：{VAL_QUESTIONS_FILE}")
    print(f"   答案：{VAL_ANSWERS_FILE}")
    print(f"   圖片：{IMAGES_DIR}")

if __name__ == "__main__":
    main()
