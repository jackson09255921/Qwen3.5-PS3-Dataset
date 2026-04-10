import json
import base64
import os
import time
from io import BytesIO
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from datasets import load_dataset, get_dataset_config_names

# ---------- 1. 配置 ----------
BASE_DIR = Path("../M3IT").resolve()
IMAGE_DIR = BASE_DIR / "images"
OUTPUT_FILE = BASE_DIR / "m3it_train.jsonl"

BASE_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# 只取 20% 的訓練資料（與原 notebook 一致）
SAMPLE_RATIO = 20

# 排除重複或低品質子集
EXCLUDE_CONFIGS = {"iqa", "iqa-rephrased", "mmchat"}

# ---------- 2. 圖片存檔 ----------
def save_b64_image(b64_str, save_path):
    if not isinstance(b64_str, str) or not b64_str:
        return False
    try:
        img_bytes = base64.b64decode(b64_str)
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        img.save(save_path, format="JPEG", quality=90, optimize=True)
        return True
    except Exception:
        return False

# ---------- 3. 載入有效 configs ----------
def get_valid_configs():
    print("🔍 掃描 M3IT 可用子集（單圖）...")
    all_configs = get_dataset_config_names("MMInstruction/M3IT")
    ok_configs = []
    for cfg in all_configs:
        if cfg in EXCLUDE_CONFIGS:
            continue
        try:
            ds = load_dataset("MMInstruction/M3IT", cfg, split="train",
                              streaming=True, trust_remote_code=True)
            ex = next(iter(ds.take(1)))
            val = ex.get("image_base64_str", [])
            if isinstance(val, list) and len(val) == 1:
                ok_configs.append(cfg)
            # 多圖跳過
        except Exception as e:
            print(f"  [SKIP] {cfg}: {e}")
    print(f"  有效子集 {len(ok_configs)} 個: {ok_configs}")
    return ok_configs

# ---------- 4. 帶 retry 的 load_dataset ----------
def load_with_retry(cfg, split, max_retries=5):
    for attempt in range(1, max_retries + 1):
        try:
            return load_dataset("MMInstruction/M3IT", cfg, split=split,
                                trust_remote_code=True)
        except Exception as e:
            print(f"  [retry {attempt}/{max_retries}] {cfg}: {e}")
            if attempt == max_retries:
                raise
            time.sleep(10)

# ---------- 5. 主程序 ----------
def main():
    print("🚀 啟動 M3IT 資料集處理程序...")

    ok_configs = get_valid_configs()
    global_i = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
        for cfg in tqdm(ok_configs, desc="子集進度"):
            split = f"train[:{SAMPLE_RATIO}%]"
            try:
                ds = load_with_retry(cfg, split)
            except Exception as e:
                print(f"  ❌ 跳過 {cfg}: {e}")
                continue

            cfg_count = 0
            for example in ds:
                img_field = example.get("image_base64_str")
                if isinstance(img_field, list):
                    b64_str = img_field[0] if img_field else None
                else:
                    b64_str = img_field

                if not b64_str:
                    continue

                img_filename = f"{global_i:08d}.jpg"
                save_path = IMAGE_DIR / img_filename

                if not save_path.exists():
                    if not save_b64_image(b64_str, save_path):
                        continue

                instr = (example.get("instruction") or "").strip()
                inp = (example.get("inputs") or "").strip()
                out = (example.get("outputs") or "").strip()

                if not out:
                    continue

                human_text = (instr + "\n" + inp).strip() if inp else instr
                human_text = f"<image>\n{human_text}"

                formatted = {
                    "id": f"m3it_{cfg}_{global_i}",
                    "image": img_filename,
                    "source": cfg,
                    "conversations": [
                        {"from": "human", "value": human_text},
                        {"from": "gpt", "value": out},
                    ],
                }

                f_out.write(json.dumps(formatted, ensure_ascii=False) + "\n")
                global_i += 1
                cfg_count += 1

            print(f"  ✅ {cfg}: {cfg_count} 筆")

    print(f"\n🎉 M3IT 處理完成！")
    print(f"  總樣本數: {global_i} 筆")
    print(f"  圖片目錄: {IMAGE_DIR}")
    print(f"  輸出檔案: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
