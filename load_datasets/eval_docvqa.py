import json
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset

# ---------- 路徑（對齊 default.yaml）----------
BASE_DIR = Path("/home/fireblue/datasets/eval/docvqa")
IMAGES_DIR = BASE_DIR / "images" / "documents"   # eval script: os.path.join(image_dir, "documents/xxx.png")
OUTPUT_FILE = BASE_DIR / "docvqa.jsonl"           # yaml: docvqa.jsonl，格式 {"data": [...]}

BASE_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# ---------- 主程序 ----------
def main():
    print("🚀 下載 DocVQA validation split...")
    ds = load_dataset("HuggingFaceM4/DocumentVQA", split="validation")

    records = []
    for item in tqdm(ds, desc="處理"):
        pil_img = item.get("image")
        question_id = item.get("questionId")
        question = str(item.get("question", "")).strip()
        answers = item.get("answers", [])
        ucsf_id = str(item.get("ucsf_document_id", "")).strip()
        page_no = str(item.get("ucsf_document_page_no", "1")).strip()

        if not pil_img or not question_id or not question:
            continue

        if isinstance(answers, str):
            answers = [answers]

        # 圖片命名：{ucsf_document_id}_{page_no}.png（官方 DocVQA 命名慣例）
        img_name = f"{ucsf_id}_{page_no}.png"
        save_path = IMAGES_DIR / img_name

        if not save_path.exists():
            try:
                pil_img.convert("RGB").save(save_path)
            except Exception:
                continue

        # eval script: inst["image"] = "documents/xxx.png"
        #              img_path = os.path.join(image_dir, inst["image"])
        #              → image_dir/documents/xxx.png  ✓
        records.append({
            "questionId": question_id,
            "question": question,
            "answers": answers,
            "image": f"documents/{img_name}",
        })

    # DocVQA eval script: raw = io.load(data_path) → raw["data"]
    output = {"data": records}
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"✅ 完成：{len(records)} 筆 → {OUTPUT_FILE}")
    print(f"   圖片：{IMAGES_DIR}")

if __name__ == "__main__":
    main()
