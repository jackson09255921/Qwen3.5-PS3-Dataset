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

# 4 種子集 (referring_qa 含座標，其餘為純對話)
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

# ---------- 3. 找圖片路徑 ----------
def find_image(image_id):
    """在 images/ 和 images2/ 中尋找對應 image_id 的圖片"""
    fname = f"{int(image_id):012d}.jpg"
    for sub in ["images", "images2"]:
        p = IMAGES_DIR / sub / fname
        if p.exists():
            return p
    # fallback: 非零填充
    for sub in ["images", "images2"]:
        for p in (IMAGES_DIR / sub).glob(f"*{int(image_id)}*.jpg"):
            return p
    return None

# ---------- 4. 座標轉換（referring_qa 使用）----------
BBOX_PATTERN = re.compile(r"\[([0-9\.]+),\s*([0-9\.]+),\s*([0-9\.]+),\s*([0-9\.]+)\]")

def extract_bbox_pixel(text, img_path):
    """
    從對話文字中提取座標，並轉換為 absolute pixel XYXY。
    支援 [0,1] 和 [0,1000] 兩種 normalized 格式。
    """
    match = BBOX_PATTERN.search(text)
    if not match:
        return None
    try:
        vals = [float(x) for x in match.groups()]
        with Image.open(img_path) as img:
            W, H = img.size
        # 判斷是否為 [0,1000] 格式
        factor = 1000.0 if any(v > 1.1 for v in vals) else 1.0
        x1 = round(vals[0] / factor * W, 2)
        y1 = round(vals[1] / factor * H, 2)
        x2 = round(vals[2] / factor * W, 2)
        y2 = round(vals[3] / factor * H, 2)
        return [x1, y1, x2, y2]
    except Exception:
        return None

# ---------- 5. 處理各子集 ----------
def process_subset(subset_name):
    json_path = BASE_DIR / subset_name / f"{subset_name}.json"
    if not json_path.exists():
        # 嘗試遞迴搜尋
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
    skip_count = 0

    with open(output_file, "w", encoding="utf-8") as f_out:
        for item in tqdm(data, desc=f"處理 {subset_name}"):
            image_id = item.get("image_id")
            convs = item.get("conversations", [])

            if not image_id or not convs:
                skip_count += 1
                continue

            img_path = find_image(image_id)
            if not img_path:
                skip_count += 1
                continue

            # 相對圖片路徑（從 SVIT base 往下）
            try:
                rel_img = img_path.relative_to(BASE_DIR)
            except ValueError:
                rel_img = img_path

            # 確保 human 發言有 <image> 標籤
            if convs[0]["from"] == "human" and "<image>" not in convs[0]["value"]:
                convs[0]["value"] = "<image>\n" + convs[0]["value"]

            formatted = {
                "id": f"svit_{subset_name}_{valid_count}",
                "image": str(rel_img),
                "conversations": convs,
            }

            # referring_qa：提取 bbox 轉為 absolute pixel XYXY
            if is_referring:
                full_text = " ".join(c["value"] for c in convs)
                bbox = extract_bbox_pixel(full_text, img_path)
                if bbox:
                    formatted["box"] = bbox
                    # 將 GPT 回答統一改為 absolute pixel 格式
                    formatted["conversations"][-1]["value"] = (
                        f"[{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]"
                    )
                else:
                    skip_count += 1
                    continue

            f_out.write(json.dumps(formatted, ensure_ascii=False) + "\n")
            valid_count += 1

    print(f"  ✅ {subset_name}: {valid_count} 筆，跳過 {skip_count} 筆 → {output_file.name}")

# ---------- 6. 主程序 ----------
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
