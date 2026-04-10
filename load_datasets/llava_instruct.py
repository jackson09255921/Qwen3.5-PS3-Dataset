import os
import json
import subprocess
from pathlib import Path
from tqdm import tqdm

def run_command(cmd):
    """執行 Shell 指令"""
    print(f"執行指令: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def prepare_pipeline():
    # ---------- 1. 環境設定 ----------
    base_path = Path('../llava_instruct')
    image_path = base_path / 'images' / 'train2017'
    json_path = Path('llava_instruct_150k.json')
    output_single_path = base_path / 'llava_single.jsonl'
    
    base_path.mkdir(parents=True, exist_ok=True)
    image_path.mkdir(parents=True, exist_ok=True)

    # ---------- 2. 下載 COCO 圖片 ----------
    if not list(image_path.glob('*.jpg')):
        print("--- 正在下載 COCO train2017 (~19GB) ---")
        download_dir = image_path.parent
        # 使用 wget 下載到正確位置
        run_command(f"wget -c http://images.cocodataset.org/zips/train2017.zip -P {download_dir}")
        run_command(f"unzip -q {download_dir}/train2017.zip -d {download_dir}")
        run_command(f"rm {download_dir}/train2017.zip")
        print("圖片下載並解壓完成！")
    else:
        print(f"圖片已存在: {len(list(image_path.glob('*.jpg')))} 張")

    # ---------- 3. 下載 LLaVA 標註資料 ----------
    if not json_path.exists():
        print("--- 正在從 HuggingFace 下載 LLaVA-Instruct JSON ---")
        run_command("wget https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/llava_instruct_150k.json")
    else:
        print("標註資料檔案已存在。")

    # ---------- 4. 格式轉換 (反推模型預期格式) ----------
    print("--- 開始格式化資料為 Model Pipeline 預期格式 ---")
    
    with open(json_path, 'r') as f:
        raw_data = json.load(f)

    single_turn_count = 0
    with open(output_single_path, 'w', encoding='utf-8') as f_out:
        for item in tqdm(raw_data, desc="處理進度"):
            # 只抓取 Single-turn (一問一答) 確保訓練穩定
            if len(item["conversations"]) == 2:
                
                # 關鍵修正：確保 image 欄位是純字串 (String)
                image_val = item.get("image", item.get("images", ""))
                if isinstance(image_val, list):
                    image_val = image_val[0] if len(image_val) > 0 else ""
                
                # 建立符合 LazySupervisedDataset 預期的結構
                formatted_item = {
                    "id": item.get("id", f"sample_{single_turn_count}"),
                    "image": str(image_val),  # 強制字串化，解決 FileNotFoundError: .../0
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
                
                # 確保 <image> 標籤在正確位置
                conv_text = formatted_item["conversations"][0]["value"]
                if "<image>" not in conv_text:
                    formatted_item["conversations"][0]["value"] = "<image>\n" + conv_text
                
                # 寫入 JSONL
                f_out.write(json.dumps(formatted_item, ensure_ascii=False) + '\n')
                single_turn_count += 1

    print(f"\n--- Pipeline 執行完畢 ---")
    print(f"最終資料集筆數: {single_turn_count}")
    print(f"訓練用 JSONL 路徑: {output_single_path}")
    print(f"圖片目錄路徑: {image_path}")
    
    # 打印一筆範例確認格式
    with open(output_single_path, 'r') as f:
        print("\n範例資料格式檢查:")
        print(f.readline())

if __name__ == "__main__":
    prepare_pipeline()