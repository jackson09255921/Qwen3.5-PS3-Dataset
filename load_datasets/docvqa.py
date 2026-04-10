import json
import io
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from PIL import Image

# ---------- 1. 配置與路徑 ----------
BASE_DIR = Path("../DocVQA").resolve()
BASE_DIR.mkdir(parents=True, exist_ok=True)

# 只處理 train split（dataset_mixture 只用 docvqa_train.jsonl）
SPLITS = ["train"]

# ---------- 2. 主處理程序 ----------
def process_all_splits():
    print("🚀 啟動 DocVQA 下載與解碼程序...")

    for split in SPLITS:
        print("\n" + "=" * 50)
        print(f"🎯 正在處理 Split: [{split.upper()}]")
        print("=" * 50)

        try:
            ds = load_dataset("lmms-lab/DocVQA", "DocVQA", split=split)
        except Exception as e:
            print(f"❌ 載入 {split} 失敗：{e}")
            continue

        split_images_dir = BASE_DIR / "images" / split
        split_images_dir.mkdir(parents=True, exist_ok=True)

        output_file = BASE_DIR / f"docvqa_{split}.jsonl"
        valid_count = 0

        with open(output_file, "w", encoding="utf-8") as f_out:
            for item in tqdm(ds, desc=f"解碼與轉換 {split}"):
                # lmms-lab/DocVQA 欄位：questionId, question, answers, image, docId
                raw_image_data = item.get("image")
                question = item.get("question", "").strip()

                # answers 可能是 list 或 string
                raw_answers = item.get("answers", item.get("answer", ""))
                if isinstance(raw_answers, list) and len(raw_answers) > 0:
                    answer = str(raw_answers[0]).strip()
                else:
                    answer = str(raw_answers).strip()

                if not raw_image_data or not question or not answer:
                    continue

                # ---------- 決定圖片檔名 ----------
                img_name = item.get("image_filename") or item.get("img_filename") or item.get("docId")
                if img_name:
                    img_name = str(img_name).strip()
                    # 補副檔名
                    if not Path(img_name).suffix:
                        img_name = img_name + ".png"
                else:
                    img_name = f"docvqa_{split}_{valid_count}.png"

                # ---------- 還原 PIL Image ----------
                try:
                    if isinstance(raw_image_data, list):
                        pil_img = Image.open(io.BytesIO(bytes(raw_image_data)))
                    elif isinstance(raw_image_data, bytes):
                        pil_img = Image.open(io.BytesIO(raw_image_data))
                    else:
                        pil_img = raw_image_data  # 已是 PIL Image
                except Exception:
                    continue

                # ---------- 儲存圖片 ----------
                save_path = split_images_dir / img_name
                if not save_path.exists():
                    try:
                        save_path.parent.mkdir(parents=True, exist_ok=True)
                        pil_img.convert("RGB").save(save_path)
                    except Exception:
                        continue

                # ---------- 格式化 JSONL ----------
                relative_img_path = f"images/{split}/{img_name}"

                formatted_data = {
                    "id": f"docvqa_{split}_{valid_count}",
                    "image": relative_img_path,
                    "conversations": [
                        {
                            "from": "human",
                            "value": f"<image>\n{question}"
                        },
                        {
                            "from": "gpt",
                            "value": answer
                        }
                    ]
                }

                f_out.write(json.dumps(formatted_data, ensure_ascii=False) + "\n")
                valid_count += 1

        print(f"✅ [{split.upper()}] 處理完成！共儲存 {valid_count} 筆資料。")
        print(f"📂 圖片路徑: {split_images_dir}")
        print(f"📄 標註檔案: {output_file.name}")

        # 預覽第一筆
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                print("\n🔍 預覽第一筆:")
                print(f.readline())
        except Exception:
            pass

if __name__ == "__main__":
    process_all_splits()
