import json
import re
import zipfile
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from huggingface_hub import hf_hub_download

# ---------- 1. 配置 ----------
BASE_DIR = Path("../SVIT").resolve()
IMAGES_DIR = BASE_DIR / "images"
BASE_DIR.mkdir(parents=True, exist_ok=True)

REPO_ID = "BAAI/SVIT"

DATA_SUBSETS = [
    "data/complex_reasoning.zip",
    "data/conversation.zip",
    "data/detail_description.zip",
    "data/referring_qa.zip",
]
IMAGE_ZIPS = [
    "raw/images.zip",
    "raw/images2.zip",
]

# ---------- 2. 下載與解壓 ----------
def setup():
    print("📥 正在下載 SVIT 標註資料...")
    for rel_path in DATA_SUBSETS:
        subset = Path(rel_path).stem
        out_dir = BASE_DIR / subset
        json_path = out_dir / f"{subset}.json"
        if json_path.exists():
            print(f"  ✅ {subset} 已存在，跳過")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        local_zip = hf_hub_download(repo_id=REPO_ID, filename=rel_path, repo_type="dataset")
        with zipfile.ZipFile(local_zip, "r") as zf:
            for member in tqdm(zf.infolist(), desc=f"  解壓 {subset}", unit="file"):
                zf.extract(member, out_dir)
        print(f"  ✅ {subset} 解壓完成")

    print("\n📥 正在下載 SVIT 圖片...")
    for rel_path in IMAGE_ZIPS:
        name = Path(rel_path).stem  # images / images2
        out_dir = IMAGES_DIR / name
        if out_dir.exists() and any(out_dir.iterdir()):
            print(f"  ✅ {name} 已存在，跳過")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        local_zip = hf_hub_download(repo_id=REPO_ID, filename=rel_path, repo_type="dataset")
        with zipfile.ZipFile(local_zip, "r") as zf:
            for member in tqdm(zf.infolist(), desc=f"  解壓 {name}", unit="file"):
                zf.extract(member, out_dir)
        print(f"  ✅ {name} 解壓完成")

    merge_vg_images()

# ---------- 3. 合併圖片到 images/ ----------
# 解壓後結構：images/images/VG_100K/ + images/images2/VG_100K_2/
# 全部搬到 images/ 直接放，結果為 images/1.jpg、images/2.jpg ...
VG_SOURCE_DIRS = [
    IMAGES_DIR / "images" / "VG_100K",
    IMAGES_DIR / "images2" / "VG_100K_2",
    IMAGES_DIR / "images" / "VG_100K_2",
    IMAGES_DIR / "images2" / "VG_100K",
]

def merge_vg_images():
    """將 VG_100K / VG_100K_2 直接移到 images/ 底下"""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    existing = sum(1 for _ in IMAGES_DIR.glob("*.jpg"))
    if existing > 0:
        print(f"  ✅ images/ 已有 {existing} 張圖，跳過合併")
        return

    total = 0
    for src_dir in VG_SOURCE_DIRS:
        if not src_dir.exists():
            continue
        files = list(src_dir.glob("*.jpg"))
        print(f"  📂 合併 {src_dir.name}: {len(files)} 張...")
        for src in tqdm(files, desc=f"  mv {src_dir.name}", leave=False):
            dst = IMAGES_DIR / src.name
            if not dst.exists():
                src.rename(dst)
            total += 1
    print(f"  ✅ 合併完成，共 {total} 張 → {IMAGES_DIR}")

def find_image(image_id):
    fname = f"{int(image_id)}.jpg"
    p = IMAGES_DIR / fname
    if p.exists():
        return p
    # 合併前的 fallback
    for d in VG_SOURCE_DIRS:
        p = d / fname
        if p.exists():
            return p
    return None

# ---------- 4. 文字清理 ----------
# SVIT 標記格式：<st>entity_name<ed> [x1, y1, x2, y2]
# 訓練時不使用這些特殊 token，移除標記只保留 entity 名稱與去掉座標
ST_TAG_PATTERN = re.compile(r"<st>(.*?)<ed>\s*\[[0-9\.,\s]+\]")
BBOX_PATTERN = re.compile(r"\[([0-9\.]+),\s*([0-9\.]+),\s*([0-9\.]+),\s*([0-9\.]+)\]")

def clean_text(text):
    """移除 <st>...<ed> 標記和嵌入座標，只保留 entity 名稱"""
    return ST_TAG_PATTERN.sub(r"\1", text).strip()

# ---------- 5. 座標轉換（referring_qa 使用）----------
def extract_first_bbox_pixel(text, img_path):
    """
    從文字中提取第一個 bbox，normalized [0,1] → absolute pixel XYXY
    """
    match = BBOX_PATTERN.search(text)
    if not match:
        return None
    try:
        vals = [float(x) for x in match.groups()]
        with Image.open(img_path) as img:
            W, H = img.size
        x1 = round(vals[0] * W, 2)
        y1 = round(vals[1] * H, 2)
        x2 = round(vals[2] * W, 2)
        y2 = round(vals[3] * H, 2)
        return [x1, y1, x2, y2]
    except Exception:
        return None

# ---------- 6. 對話結構攤平 ----------
def flatten_conversations(raw_conversations, add_image_tag=True):
    """
    SVIT 結構：conversations = [{content: [{from,value},{from,value}]}, ...]
    攤平為：[{from, value}, {from, value}, ...]
    第一個 human 發言加入 <image> 標籤
    """
    turns = []
    for exchange in raw_conversations:
        content = exchange.get("content", [])
        for msg in content:
            role = msg.get("from", "")
            value = msg.get("value", "").strip()
            if role == "user":
                role = "human"
            elif role != "gpt":
                continue
            turns.append({"from": role, "value": value})

    if not turns:
        return None

    # 確保從 human 開始
    if turns[0]["from"] != "human":
        return None

    # 第一個 human 發言加 <image> 標籤
    if add_image_tag and "<image>" not in turns[0]["value"]:
        turns[0]["value"] = "<image>\n" + turns[0]["value"]

    return turns

# ---------- 7. 處理各子集 ----------
def process_subset(subset_name):
    json_path = BASE_DIR / subset_name / f"{subset_name}.json"
    if not json_path.exists():
        found = list((BASE_DIR / subset_name).rglob("*.json"))
        if not found:
            print(f"❌ 找不到 {subset_name} 的 JSON 檔案，跳過")
            return
        json_path = found[0]

    output_file = BASE_DIR / f"svit_{subset_name}.jsonl"
    is_referring = subset_name == "referring_qa"

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    valid_count = 0
    skip_no_image = 0
    skip_no_conv = 0
    skip_no_bbox = 0

    with open(output_file, "w", encoding="utf-8") as f_out:
        for item in tqdm(data, desc=f"處理 {subset_name}"):
            image_id = item.get("image_id")
            raw_convs = item.get("conversations", [])

            if not image_id or not raw_convs:
                skip_no_conv += 1
                continue

            img_path = find_image(image_id)
            if not img_path:
                skip_no_image += 1
                continue

            # 相對路徑（從 SVIT base 往下，訓練時 image_folder=~/datasets/SVIT）
            rel_img = f"images/{int(image_id)}.jpg"

            # 攤平對話並清理 <st>...<ed> 標記
            turns = flatten_conversations(raw_convs)
            if not turns:
                skip_no_conv += 1
                continue

            for t in turns:
                t["value"] = clean_text(t["value"])

            formatted = {
                "id": f"svit_{subset_name}_{valid_count}",
                "image": rel_img,
                "conversations": turns,
            }

            # referring_qa：從第一個 human 原始文字提取 bbox → absolute pixel XYXY
            if is_referring:
                # 在清理前取得原始第一個 human 發言做座標提取
                first_human_raw = raw_convs[0]["content"][0]["value"] if raw_convs[0].get("content") else ""
                bbox = extract_first_bbox_pixel(first_human_raw, img_path)
                if not bbox:
                    skip_no_bbox += 1
                    continue
                formatted["box"] = bbox

            f_out.write(json.dumps(formatted, ensure_ascii=False) + "\n")
            valid_count += 1

    msg = f"  ✅ {subset_name}: {valid_count} 筆"
    if skip_no_image:
        msg += f"，找不到圖片: {skip_no_image}"
    if skip_no_conv:
        msg += f"，格式異常: {skip_no_conv}"
    if is_referring and skip_no_bbox:
        msg += f"，無 bbox: {skip_no_bbox}"
    print(msg + f" → {output_file.name}")

# ---------- 8. 主程序 ----------
def main():
    print("🚀 啟動 SVIT 資料集處理程序...")
    setup()

    print("\n✨ 開始轉換各子集...")
    for rel_path in DATA_SUBSETS:
        process_subset(Path(rel_path).stem)

    print("\n🎉 全部處理完畢！")
    print("輸出檔案：")
    for f in sorted(BASE_DIR.glob("svit_*.jsonl")):
        count = sum(1 for _ in open(f, encoding="utf-8"))
        print(f"  {f.name}: {count} 筆")

if __name__ == "__main__":
    main()
