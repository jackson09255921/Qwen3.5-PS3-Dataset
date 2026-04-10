import os
import json
import re
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ---------- 1. 初始化設定 ----------
BASE_DIR = Path("../ScreenSpot-v2").resolve()
IMAGES_SAVE_DIR = BASE_DIR / "images"
OUTPUT_FILE = BASE_DIR / "screenspot_v2_grounding_pixel.jsonl"

IMAGES_SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ---------- 2. 座標與文本處理函數 ----------

def convert_xywh_to_xyxy(bbox_xywh):
    """將 [x, y, w, h] 絕對像素轉換為 [x1, y1, x2, y2]"""
    try:
        x, y, w, h = [float(c) for c in bbox_xywh]
        # x1, y1, x2, y2
        return [round(x, 2), round(y, 2), round(x + w, 2), round(y + h, 2)]
    except:
        return None

def clean_instruction(text):
    """清理指令文本，確保格式正確"""
    clean = text.strip()
    # 移除多餘空白
    clean = " ".join(clean.split())
    return f"<image>\n{clean}"

# ---------- 3. 主程序執行流 ----------

def run_process_and_visualize():
    print("🚀 正在從 Hugging Face 載入 ScreenSpot-v2...")
    try:
        # 載入 dataset (自動下載圖片)
        ds = load_dataset("lmms-lab/ScreenSpot-v2", split="train") 
    except Exception as e:
        print(f"❌ 載入失敗：{e}")
        return

    valid_count = 0
    preview_data = []

    print(f"📥 正在儲存圖片並生成 JSONL...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
        for item in tqdm(ds, desc="Processing"):
            img_name = item.get("img_filename")
            pil_img = item.get("image") # 這是 HF 載入的 PIL Image 對象
            raw_bbox = item.get("bbox")
            instruction = item.get("instruction", "Locate the element.")
            
            if not img_name or not pil_img or not raw_bbox:
                continue
            
            # 1. 儲存圖片到本地 (如果不存在)
            save_path = IMAGES_SAVE_DIR / img_name
            if not save_path.exists():
                save_path.parent.mkdir(parents=True, exist_ok=True)
                pil_img.save(save_path)

            # 2. 座標轉換 [x, y, w, h] -> [x1, y1, x2, y2]
            bbox_xyxy = convert_xywh_to_xyxy(raw_bbox)
            if not bbox_xyxy:
                continue

            # 3. 建立訓練格式資料
            formatted_data = {
                "id": f"ss_v2_{valid_count}",
                "image": str(img_name),
                "box": bbox_xyxy, # 絕對像素格式
                "conversations": [
                    {"from": "human", "value": clean_instruction(instruction)},
                    {"from": "gpt", "value": f"Located at [{bbox_xyxy[0]}, {bbox_xyxy[1]}, {bbox_xyxy[2]}, {bbox_xyxy[3]}]"}
                ]
            }

            f_out.write(json.dumps(formatted_data, ensure_ascii=False) + "\n")
            
            # 收集前 10 筆供稍後檢查
            if valid_count < 10:
                preview_data.append((save_path, bbox_xyxy, instruction))
            
            valid_count += 1

    print(f"\n✅ 處理完成！總計 {valid_count} 筆資料已存入 {OUTPUT_FILE.name}")
    
    # ---------- 4. 互動式檢查 (按 Space 切換) ----------
    print("\n📺 正在啟動肉眼檢查 (前 10 筆)...")
    print("👉 操作提示：點擊視窗後按【空白鍵】切換下一張")

    for i, (img_path, box, prompt) in enumerate(preview_data):
        img = Image.open(img_path).convert("RGB")
        x1, y1, x2, y2 = box

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(img)
        
        # 畫出紅色 Bbox
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1, 
            linewidth=3, edgecolor='#FF0000', facecolor='none'
        )
        ax.add_patch(rect)
        
        plt.title(f"[ScreenSpot-v2] {i+1}/10\nPrompt: {prompt}", loc='left', fontsize=10, pad=20)
        ax.axis('off')
        
        plt.draw()
        # 等待按鍵 (plt.waitforbuttonpress() True 為鍵盤, False 為滑鼠)
        while not plt.waitforbuttonpress():
            pass
        plt.close()

if __name__ == "__main__":
    run_process_and_visualize()
    print("\n🎉 全部任務執行完畢！")