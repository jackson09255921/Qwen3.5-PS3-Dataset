import json
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset

# ---------- 路徑（對齊 default.yaml）----------
BASE_DIR   = Path("/home/fireblue/datasets/eval/mathvista")
IMAGES_DIR = BASE_DIR / "images"
BASE_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = {
    "testmini": BASE_DIR / "mathvista_testmini.jsonl",
    "test":     BASE_DIR / "mathvista_test.jsonl",
}

# ---------- 主程序 ----------
def main():
    print("🚀 下載 MathVista dataset...")
    ds_all = load_dataset("AI4Math/MathVista")

    for split, output_file in SPLITS.items():
        if split not in ds_all:
            print(f"  ⚠️  split '{split}' 不存在，跳過")
            continue

        ds = ds_all[split]
        print(f"\n🎯 處理 split: {split} ({len(ds)} 筆)")

        with open(output_file, "w", encoding="utf-8") as f_out:
            for item in tqdm(ds, desc=split):
                pid = item["pid"]
                pil_img = item.get("decoded_image")

                # 儲存圖片
                img_name = f"{pid}.jpg"
                save_path = IMAGES_DIR / img_name
                if not save_path.exists() and pil_img is not None:
                    try:
                        pil_img.convert("RGB").save(save_path)
                    except Exception:
                        pass

                # 序列化所有文字欄位，image 改為本地路徑
                record = {k: v for k, v in item.items() if k != "decoded_image"}
                record["image"] = str(save_path)  # eval script fallback: Image.open(img_path)

                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

        count = sum(1 for _ in open(output_file))
        print(f"  ✅ {split}: {count} 筆 → {output_file.name}")

    print(f"\n🎉 完成！圖片：{IMAGES_DIR}")

if __name__ == "__main__":
    main()
