import os
import json
import subprocess
from pathlib import Path
from tqdm import tqdm

def run_command(cmd):
    """執行 Shell 指令"""
    print(f"執行指令: {cmd}")
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"指令執行失敗: {e}")

def prepare_step1_pipeline():
    # ---------- 1. 路徑設定 (往外放一層 ../) ----------
    # 設定父目錄路徑
    parent_dir = Path('../datasets').resolve() # 建議建立一個 datasets 資料夾統一管理
    parent_dir.mkdir(parents=True, exist_ok=True)

    base_path = parent_dir / 'LLaVA-CC3M-Pretrain-595K'
    json_input_path = base_path / 'chat.json'
    image_zip = base_path / 'images.zip'
    image_folder = base_path / 'images'
    
    # 輸出的 JSONL 也放在外面，方便 Stage 1 訓練讀取
    output_jsonl_path = parent_dir / 'cc3m_align_single.jsonl'

    # ---------- 2. 環境與資料下載 ----------
    # Git Clone 到父目錄
    if not base_path.exists():
        print(f"--- 正在 Clone 資料集至 {parent_dir} ---")
        run_command(f"git clone https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K {base_path}")
    
    # 解壓縮圖片
    if not image_folder.exists():
        print("--- 正在解壓縮 CC3M 圖片 ---")
        image_folder.mkdir(parents=True, exist_ok=True)
        # 解壓到指定的外部目錄
        run_command(f"unzip -q {image_zip} -d {image_folder}")
        print("解壓完成。")
    else:
        print(f"圖片目錄已存在: {image_folder}")

    # ---------- 3. 格式轉換 (Step 1: Alignment) ----------
    print(f"\n--- 開始處理 CC3M 資料 (目標: Projector Alignment) ---")
    
    if not json_input_path.exists():
        print(f"錯誤：找不到標註檔案 {json_input_path}")
        return

    with open(json_input_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    processed_count = 0
    with open(output_jsonl_path, 'w', encoding='utf-8') as f_out:
        for item in tqdm(raw_data, desc="轉換進度"):
            # 確保資料結構完整 (需包含 Human 與 GPT 對話) [cite: 1300, 1301]
            if "conversations" not in item or len(item["conversations"]) < 2:
                continue

            # 提取圖片檔名 [cite: 283]
            image_val = item.get("image", "")
            
            # 建立符合 VILA-HD Pipeline 預期的結構 [cite: 282, 1299]
            formatted_item = {
                "id": item.get("id", f"cc3m_{processed_count}"),
                "image": str(image_val),
                "conversations": [
                    {
                        "from": "human",
                        "value": item["conversations"][0]["value"]
                    },
                    {
                        "from": "gpt",
                        "value": item["conversations"][1]["value"]
                    }
                ]
            }

            # 確保 <image> 標籤在 Human 提問的最前方 [cite: 274, 283]
            # 這是 VILA-HD 訓練 Projector 接收視覺訊號的標準格式 [cite: 1300]
            human_text = formatted_item["conversations"][0]["value"]
            if "<image>" not in human_text:
                formatted_item["conversations"][0]["value"] = "<image>\n" + human_text

            # 寫入 JSONL 檔案
            f_out.write(json.dumps(formatted_item, ensure_ascii=False) + '\n')
            processed_count += 1

    print(f"\n--- CC3M Step 1 處理成功 ---")
    print(f"有效樣本總數: {processed_count}")
    print(f">>> Step 1 訓練 JSONL 路徑: {output_jsonl_path}")
    print(f">>> 圖片目錄路徑: {image_folder}")

    # 檢查一筆範例確認格式
    try:
        with open(output_jsonl_path, 'r') as f:
            print("\n範例格式檢查 (JSONL):")
            print(f.readline())
    except Exception as e:
        print(f"檢查檔案時發生錯誤: {e}")

if __name__ == "__main__":
    prepare_step1_pipeline()