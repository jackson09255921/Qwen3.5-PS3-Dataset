import os
import json
import requests
import zipfile
import shutil
from pathlib import Path
from tqdm import tqdm

# ---------- 1. 配置與路徑 ----------
# 建議路徑：/home/fireblue/datasets/ShareGPT4V
BASE_DIR = Path("/home/fireblue/datasets/ShareGPT4V").resolve()
IMAGES_SAVE_DIR = BASE_DIR / "images"

# 確保資料夾存在
IMAGES_SAVE_DIR.mkdir(parents=True, exist_ok=True)
BASE_DIR.mkdir(parents=True, exist_ok=True)

# 🚀 標註檔 (DataOptim) 與 圖片 (COCO 2017 官方)
HF_TEXT_URL = "https://huggingface.co/datasets/BAAI/DataOptim/resolve/main/data/sharegpt4v.json"
COCO_2017_IMG_URL = "http://images.cocodataset.org/zips/train2017.zip"

RAW_JSON = BASE_DIR / "sharegpt4v_raw.json"
OUTPUT_JSONL = BASE_DIR / "sharegpt4v_processed.jsonl"

# ---------- 2. 下載標註 JSON (Step 0) ----------
def download_text_data():
    print("\n" + "="*50)
    print("🚀 [Step 0] 下載 ShareGPT4V 標註 JSON...")
    print("="*50)
    
    if RAW_JSON.exists():
        print(f"✅ 標註檔已存在: {RAW_JSON}")
        return

    try:
        r = requests.get(HF_TEXT_URL, stream=True, timeout=60)
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        
        with open(RAW_JSON, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, unit_divisor=1024, desc="⬇️ Downloading JSON"
        ) as pbar:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        print("✅ 標註檔下載完成！")
    except Exception as e:
        print(f"❌ 下載標註檔失敗: {e}")

# ---------- 3. 下載與解壓 2017 圖片 (Step 1) ----------
def download_and_extract_2017():
    print("\n" + "="*50)
    print("🚀 [Step 1] 下載與準備 COCO 2017 圖片 (約 18GB)...")
    print("="*50)
    
    zip_path = IMAGES_SAVE_DIR / "train2017.zip"
    target_dir = IMAGES_SAVE_DIR / "coco_2017" # 這是 JSON 期待的資料夾名稱

    # 如果已經有圖，就跳過
    if target_dir.exists() and any(target_dir.iterdir()):
        print(f"✅ {target_dir.name} 已準備就緒，跳過下載。")
        return

    # A. 下載
    if not zip_path.exists():
        try:
            r = requests.get(COCO_2017_IMG_URL, stream=True, timeout=120)
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            
            with open(zip_path, "wb") as f, tqdm(
                total=total, unit="B", unit_scale=True, unit_divisor=1024, desc="⬇️ Downloading train2017.zip"
            ) as pbar:
                for chunk in r.iter_content(chunk_size=16384):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        except Exception as e:
            print(f"❌ 圖片下載失敗: {e}")
            return

    # B. 解壓並重新命名以對齊路徑
    print("📦 正在解壓縮 (2017 包較大，請耐心等候)...")
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(IMAGES_SAVE_DIR)
        
        # COCO 解壓後通常叫 "train2017"，我們把它改名為 "coco_2017" 來對齊 JSON
        extracted_folder = IMAGES_SAVE_DIR / "train2017"
        if extracted_folder.exists():
            if target_dir.exists():
                shutil.rmtree(target_dir) # 清理舊的空資料夾
            extracted_folder.rename(target_dir)
            
        zip_path.unlink() # 刪除 zip 節省硬碟
        print(f"✅ 圖片解壓完成，已自動對齊至: {target_dir}")
    except Exception as e:
        print(f"❌ 解壓失敗: {e}")

# ---------- 4. 解析、驗證並生成 JSONL (Step 2) ----------
def process_data_2017():
    print("\n" + "="*50)
    print("🚀 [Step 2] 驗證路徑並生成最終 JSONL...")
    print("="*50)
    
    if not RAW_JSON.exists():
        print("❌ 錯誤：找不到標註檔。")
        return

    try:
        with open(RAW_JSON, "r", encoding="utf-8") as f:
            data_list = json.load(f)
    except Exception as e:
        print(f"❌ 讀取 JSON 失敗: {e}")
        return

    valid_count = 0
    missing_images = 0
    first_sample = None

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f_out:
        for item in tqdm(data_list, desc="🔎 檢查圖片存量"):
            img_rel_path = item.get("image")
            
            if img_rel_path:
                # 此時 img_rel_path 應該是 "coco_2017/000000000009.jpg"
                full_path = IMAGES_SAVE_DIR / img_rel_path
                
                if not full_path.exists():
                    missing_images += 1
                    continue

            # 轉為 JSONL 格式寫入
            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
            
            if first_sample is None:
                first_sample = json.dumps(item, ensure_ascii=False)
            valid_count += 1

    print("\n" + "="*50)
    print("🎉 Stage 2 語言底料準備完畢！")
    print(f"✅ 成功對齊圖片: {valid_count} 筆")
    print(f"⚠️ 圖片缺失: {missing_images} 筆")
    print(f"📄 輸出檔案: {OUTPUT_JSONL}")
    if first_sample:
        print("\n🔍 預覽 (路徑已確認有效):")
        print(first_sample)
    print("="*50)

if __name__ == "__main__":
    download_text_data()
    download_and_extract_2017()
    process_data_2017()