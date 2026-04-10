import os
import json
import requests
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------- 1. 配置設定 ----------
FOLDER = Path("../GRIT").resolve()
IMAGE_FOLDER = FOLDER / "images"
JSONL_OUTPUT = FOLDER / "grit_stage1_pixel.jsonl"

IMAGE_FOLDER.mkdir(parents=True, exist_ok=True)

# 下載參數
MAX_WORKERS = 20  # 下載線程數
TIMEOUT = (3, 7)  # (connect, read)
IMAGE_WIDTH_PADDING = 8  # 檔名補零長度

# ---------- 2. 下載函數 ----------
def download_image(url, out_path, session):
    if out_path.exists():
        return True
    try:
        r = session.get(url, timeout=TIMEOUT, stream=True)
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=16384):
                if chunk:
                    f.write(chunk)
        return True
    except Exception:
        return False

# ---------- 3. 主程序 ----------
def main():
    print("--- 1. 正在從 Hugging Face 載入 GRIT Dataset ---")
    # 如果要測試，可以改用 ds = load_dataset("zzliang/GRIT", split="train").select(range(100))
    ds = load_dataset("zzliang/GRIT", split="train")

    # --- 2. 收集並排重 URL ---
    url_set = sorted(list(set(data["url"] for data in ds)))
    print(f"共 {len(url_set)} 個獨立圖片 URL")

    # --- 3. 多線程下載圖片 ---
    url2fname = {}
    session = requests.Session()
    
    print(f"--- 2. 開始下載圖片至 {IMAGE_FOLDER} ---")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_url = {}
        for idx, url in enumerate(url_set):
            fname = f"{idx+1:0{IMAGE_WIDTH_PADDING}d}.jpg"
            out_path = IMAGE_FOLDER / fname
            future_to_url[executor.submit(download_image, url, out_path, session)] = (url, fname)

        for future in tqdm(as_completed(future_to_url), total=len(url_set), desc="Downloading"):
            url, fname = future_to_url[future]
            if future.result():
                url2fname[url] = fname

    # --- 4. 轉換數據格式 (一圖一問 + 像素級座標) ---
    print(f"--- 3. 正在轉換為 Stage 1 像素級 Grounding 格式 ---")
    final_data = []
    processed_urls = set()  # 用於確保「一圖一問」
    missing_count = 0
    
    for data in tqdm(ds, desc="Formatting"):
        url = data["url"]
        
        # 核心策略 A: 確保這張圖還沒被用過 (One Image, One Box)
        if url in processed_urls or url not in url2fname:
            if url not in url2fname: missing_count += 1
            continue
        
        # 核心策略 B: 檢查是否有 Bbox
        ref_list = data.get("ref_exps", [])
        if not ref_list:
            continue

        fname = url2fname[url]
        img_path = IMAGE_FOLDER / fname

        try:
            # 核心策略 C: 讀取圖片寬高進行像素轉換
            with Image.open(img_full_path := img_path) as img:
                W, H = img.size
        except Exception:
            continue

        # 只選取第一個 ref_exp 作為代表 (避免多圖混淆)
        # GRIT: [norm_x1, norm_y1, norm_x2, norm_y2, description]
        n_x1, n_y1, n_x2, n_y2, desc = ref_list[0]

        # 核心策略 D: 轉換為絕對像素 (符合 tv_tensors XYXY 規範)
        x1 = round(float(n_x1) * W, 2)
        y1 = round(float(n_y1) * H, 2)
        x2 = round(float(n_x2) * W, 2)
        y2 = round(float(n_y2) * H, 2)

        # 建立 VILA-HD 訓練條目
        formatted_item = {
            "id": f"grit_{data['key']}",
            "image": fname,
            "box": [x1, y1, x2, y2],  # 絕對像素級座標
            "conversations": [
                {
                    "from": "human",
                    "value": f"<image>\nFind and locate: {desc}"
                },
                {
                    "from": "gpt",
                    "value": f"The object is at [{n_x1}, {n_y1}, {n_x2}, {n_y2}]."
                }
            ]
        }
        
        final_data.append(formatted_item)
        processed_urls.add(url) # 標記此圖已使用

    # --- 5. 儲存結果 ---
    print(f"--- 4. 正在寫入 JSONL ---")
    with open(JSONL_OUTPUT, "w", encoding="utf-8") as f:
        for entry in final_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\n✅ 處理完成！")
    print(f"成功處理樣本 (一圖一問): {len(final_data)} 筆")
    print(f"失敗/缺圖跳過: {missing_count} 筆")
    print(f"結果儲存於: {JSONL_OUTPUT}")

if __name__ == "__main__":
    main()