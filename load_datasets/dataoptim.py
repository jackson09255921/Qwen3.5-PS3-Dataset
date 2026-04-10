import os
import json
import re
import zipfile
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from huggingface_hub import hf_hub_download
from tqdm import tqdm

# ---------- 1. 環境設定 ----------
REPO_ID = "BAAI/DataOptim"
ROOT = Path("../DataOptim").resolve()
IMAGES_DIR = ROOT / "images"
ROOT.mkdir(parents=True, exist_ok=True)

# 下載清單 (確保包含 flickr30k)
DATA_FILES = ["data/flickr30k.json", "data/ref3reg.json", "data/ref3rec.json", "data/shikra.json"]
IMAGE_ZIPS = ["images/coco/train2014.zip", "images/flickr30k/flickr30k.zip"]

def setup_data():
    """下載資料"""
    local_jsons = []
    for remote_path in DATA_FILES:
        print(f"📥 正在下載: {remote_path}...")
        path = hf_hub_download(repo_id=REPO_ID, filename=remote_path, local_dir=ROOT, repo_type="dataset")
        local_jsons.append(Path(path))
    for remote_zip in IMAGE_ZIPS:
        print(f"📥 正在下載圖片: {remote_zip}...")
        zip_path = hf_hub_download(repo_id=REPO_ID, filename=remote_zip, local_dir=ROOT, repo_type="dataset")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(IMAGES_DIR)
    return local_jsons

# ---------- 2. 核心邏輯：座標與任務重構 ----------

def process_bbox_to_pixel(item, img_full_path):
    """紅色邏輯 A: [xmin, ymin, xmax, ymax] 比例 -> 像素"""
    pattern = r"\[([0-9\.]+),\s*([0-9\.]+),\s*([0-9\.]+),\s*([0-9\.]+)\]"
    # 嘗試從 human 或 gpt 提取原始座標
    text_to_search = item["conversations"][0]["value"] + item["conversations"][1]["value"]
    match = re.search(pattern, text_to_search)
    if not match: return None

    try:
        v_xmin, v_ymin, v_xmax, v_ymax = [float(x) for x in match.groups()]
        with Image.open(img_full_path) as img:
            W, H = img.size
        factor = 1000.0 if any(c > 1.1 for c in [v_xmin, v_ymin, v_xmax, v_ymax]) else 1.0
        x1, y1 = (v_xmin / factor) * W, (v_ymin / factor) * H
        x2, y2 = (v_xmax / factor) * W, (v_ymax / factor) * H
        return [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]
    except:
        return None

def transform_ref3reg(item):
    """
    【關鍵】將 Ref3Reg 從『生成描述』轉為『定位目標』
    原本: H: "Describe area" G: "A red dog"
    現在: H: "Locate the red dog..." G: "[x, y, x, y]"
    """
    description = item["conversations"][1]["value"].strip()
    # 建立適合 Grounding 的問題
    new_human = f"<image>\nLocate {description} and provide its coordinates, please."
    return new_human, description # 回傳描述作為參考

# ---------- 3. 執行全量轉換 ----------

def run_conversion(local_jsons):
    jsonl_files = []
    for json_path in local_jsons:
        name = json_path.stem
        output_file = ROOT / f"{name}_grounding_pixel.jsonl"
        print(f"\n✨ 正在轉換: {name} -> {output_file.name}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        valid_count = 0
        processed_images = set()

        with open(output_file, "w", encoding="utf-8") as f_out:
            for item in tqdm(raw_data, desc=f"處理 {name}"):
                img_rel_path = item["image"]
                if img_rel_path in processed_images: continue

                # 找圖片路徑
                img_full_path = IMAGES_DIR / img_rel_path
                if not img_full_path.exists():
                    for p in IMAGES_DIR.rglob(Path(img_rel_path).name):
                        img_full_path = p; break
                if not img_full_path.exists(): continue

                bbox = process_bbox_to_pixel(item, img_full_path)
                if bbox:
                    # 處理文本邏輯
                    if "ref3reg" in name:
                        h_text, _ = transform_ref3reg(item)
                        g_text = f"[{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]"
                    else:
                        # 一般 Grounding 任務 (ref3rec, shikra, flickr)
                        h_text = item["conversations"][0]["value"].replace("<image>", "").strip()
                        h_text = re.sub(r"\[[0-9\.,\s]+\]", "", h_text) # 移除問題中可能存在的座標
                        h_text = "<image>\n" + " ".join(h_text.split())
                        g_text = item["conversations"][1]["value"]

                    formatted = {
                        "id": f"do_{name}_{valid_count}",
                        "image": str(img_rel_path),
                        "box": bbox,
                        "conversations": [
                            {"from": "human", "value": h_text},
                            {"from": "gpt", "value": g_text}
                        ]
                    }
                    f_out.write(json.dumps(formatted, ensure_ascii=False) + "\n")
                    processed_images.add(img_rel_path)
                    valid_count += 1
        
        print(f"✅ 生成 {valid_count} 筆。")
        jsonl_files.append(output_file)
    return jsonl_files

# ---------- 4. 最終檢查視覺化 ----------

def run_visual_check(jsonl_files):
    for file_path in jsonl_files:
        print(f"\n📺 檢查 {file_path.name} (按空白鍵下張)")
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()[:10]
        for i, line in enumerate(lines):
            data = json.loads(line)
            img_path = IMAGES_DIR / data["image"]
            if not img_path.exists():
                for p in IMAGES_DIR.rglob(Path(data["image"]).name):
                    img_path = p; break
            
            img = Image.open(img_path).convert("RGB")
            x1, y1, x2, y2 = data["box"]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.imshow(img)
            ax.add_patch(patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=3, edgecolor='#FF0000', facecolor='none'))
            plt.title(f"[{file_path.stem}] {i+1}/10\nPrompt: {data['conversations'][0]['value'][:100]}", loc='left', fontsize=8)
            plt.axis('off')
            plt.draw()
            while not plt.waitforbuttonpress(): pass
            plt.close()

if __name__ == "__main__":
    local_jsons = setup_data()
    jsonl_files = run_conversion(local_jsons)
    run_visual_check(jsonl_files)