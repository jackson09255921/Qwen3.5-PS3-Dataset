import json
import io
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from PIL import Image

# ---------- 路徑（對齊 default.yaml）----------
BASE_DIR = Path("/home/fireblue/datasets/eval/chartqa")
IMAGES_DIR = BASE_DIR / "images"
# yaml 已更新為 test.jsonl（test split）
OUTPUT_FILE = BASE_DIR / "test.jsonl"

BASE_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# ---------- 主程序 ----------
def main():
    print("🚀 下載 ChartQA test split...")
    ds = load_dataset("ahmed-masry/ChartQA", split="test")

    valid = 0
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
        for item in tqdm(ds, desc="處理"):
            pil_img = item.get("image")
            img_name = str(item.get("imgname", "")).strip()
            question = str(item.get("query", "")).strip()
            answer = item.get("label", "")

            if not pil_img or not img_name or not question:
                continue

            if isinstance(answer, list):
                answer = str(answer[0]) if answer else ""
            else:
                answer = str(answer)

            # 儲存圖片
            save_path = IMAGES_DIR / img_name
            if not save_path.exists():
                try:
                    pil_img.convert("RGB").save(save_path)
                except Exception:
                    continue

            # eval script: inst["image"].split("/")[-1] → 直接存 imgname
            record = {
                "question": question,
                "answer": answer,
                "image": img_name,
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            valid += 1

    print(f"✅ 完成：{valid} 筆 → {OUTPUT_FILE}")
    print(f"   圖片：{IMAGES_DIR}")

if __name__ == "__main__":
    main()
