import os
import json
import io
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from PIL import Image

# ---------- 1. 配置與路徑 ----------
BASE_DIR = Path("../ChartQA").resolve()
BASE_DIR.mkdir(parents=True, exist_ok=True)

# 定義我們要抓取的分割 (Splits)
SPLITS = ["train", "val", "test"]

# ---------- 2. 主處理程序 ----------
def process_all_splits():
    print("🚀 啟動 ChartQA 全集下載與解碼程序 (免解壓縮版)...")
    
    for split in SPLITS:
        print("\n" + "="*50)
        print(f"🎯 正在處理 Split: [{split.upper()}]")
        print("="*50)
        
        try:
            ds = load_dataset("ahmed-masry/ChartQA", split=split) 
        except Exception as e:
            print(f"❌ 載入 {split} 失敗：{e}")
            continue

        # 每個 split 建立專屬的圖片資料夾 (例如: ../ChartQA/images/train/)
        split_images_dir = BASE_DIR / "images" / split
        split_images_dir.mkdir(parents=True, exist_ok=True)

        # 每個 split 輸出獨立的 JSONL (例如: chartqa_train.jsonl)
        output_file = BASE_DIR / f"chartqa_{split}.jsonl"
        valid_count = 0

        with open(output_file, "w", encoding="utf-8") as f_out:
            for item in tqdm(ds, desc=f"解碼與轉換 {split}"):
                
                raw_image_data = item.get("image")
                img_name = item.get("imgname")
                question = item.get("query", "")
                answer = item.get("label", "")

                if not raw_image_data or not img_name:
                    continue

                img_name = str(img_name).strip()

                # ---------- 還原圖片 ----------
                try:
                    if isinstance(raw_image_data, list):
                        img_bytes = bytes(raw_image_data)
                        pil_img = Image.open(io.BytesIO(img_bytes))
                    elif isinstance(raw_image_data, bytes):
                        pil_img = Image.open(io.BytesIO(raw_image_data))
                    else:
                        pil_img = raw_image_data
                except Exception as e:
                    continue

                # 儲存圖片到專屬資料夾
                save_path = split_images_dir / img_name
                if not save_path.exists():
                    try:
                        pil_img.convert('RGB').save(save_path)
                    except Exception as e:
                        continue

                # ---------- 處理文字 ----------
                if isinstance(answer, list) and len(answer) > 0:
                    answer = str(answer[0])
                else:
                    answer = str(answer)

                if not question or not answer:
                    continue

                # ---------- 格式化 JSONL ----------
                # 注意：這裡的 image 路徑加上了 split 資料夾前綴
                # 這樣 VILA 訓練時才能在 --image_folder (指向 ../ChartQA) 底下正確找到圖
                relative_img_path = f"images/{split}/{img_name}"

                formatted_data = {
                    "id": f"chartqa_{split}_{img_name.split('.')[0]}_{valid_count}",
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

if __name__ == "__main__":
    process_all_splits()