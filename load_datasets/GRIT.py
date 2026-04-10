import json
import requests
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset

# ---------- 1. 配置設定 ----------
BASE_DIR = Path("../GRIT").resolve()
IMAGE_DIR = BASE_DIR / "images"
OUTPUT_FILE = BASE_DIR / "grit_grounding_pixel.jsonl"

IMAGE_DIR.mkdir(parents=True, exist_ok=True)

MAX_WORKERS = 20
TIMEOUT = (3, 7)

# ---------- 2. 多線程圖片下載 ----------
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
    print("🚀 正在從 Hugging Face 載入 GRIT Dataset...")
    ds = load_dataset("zzliang/GRIT", split="train")

    # 收集並排重 URL
    url_set = sorted(set(data["url"] for data in ds))
    print(f"共 {len(url_set)} 個獨立圖片 URL")

    # 建立 URL -> 檔名映射
    url2fname = {url: f"{idx+1:08d}.jpg" for idx, url in enumerate(url_set)}

    # 多線程下載
    session = requests.Session()
    print(f"📥 開始下載圖片至 {IMAGE_DIR}...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(download_image, url, IMAGE_DIR / fname, session): url
            for url, fname in url2fname.items()
        }
        success = 0
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
            if future.result():
                success += 1
    print(f"✅ 下載完成：{success} / {len(url_set)}")

    # ---------- 4. 格式轉換：normalized → absolute pixel XYXY ----------
    print("✨ 正在轉換為 Grounding 格式（absolute pixel XYXY）...")
    valid_count = 0
    missing_count = 0
    processed_urls = set()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
        for data in tqdm(ds, desc="Formatting"):
            url = data["url"]

            # 一圖一問：每個 URL 只取第一個 ref_exp
            if url in processed_urls or url not in url2fname:
                if url not in url2fname:
                    missing_count += 1
                continue

            ref_list = data.get("ref_exps", [])
            if not ref_list:
                continue

            fname = url2fname[url]
            img_path = IMAGE_DIR / fname

            # 讀取圖片尺寸以換算絕對座標
            try:
                with Image.open(img_path) as img:
                    W, H = img.size
            except Exception:
                continue

            # GRIT: [norm_x1, norm_y1, norm_x2, norm_y2, description]
            n_x1, n_y1, n_x2, n_y2, desc = ref_list[0]

            # normalized [0,1] → absolute pixel XYXY
            x1 = round(float(n_x1) * W, 2)
            y1 = round(float(n_y1) * H, 2)
            x2 = round(float(n_x2) * W, 2)
            y2 = round(float(n_y2) * H, 2)

            formatted_item = {
                "id": f"grit_{data['key']}",
                "image": fname,
                "box": [x1, y1, x2, y2],
                "conversations": [
                    {
                        "from": "human",
                        "value": f"<image>\nLocate {desc} and provide its bounding box coordinates."
                    },
                    {
                        "from": "gpt",
                        "value": f"Located at [{x1}, {y1}, {x2}, {y2}]."
                    }
                ]
            }

            f_out.write(json.dumps(formatted_item, ensure_ascii=False) + "\n")
            processed_urls.add(url)
            valid_count += 1

    print(f"\n✅ 處理完成！")
    print(f"  有效樣本（一圖一問）: {valid_count} 筆")
    print(f"  缺圖跳過: {missing_count} 筆")
    print(f"📄 輸出: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
