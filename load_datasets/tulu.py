import json
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset

# ---------- 1. 配置與路徑 ----------
BASE_DIR = Path("../Tulu").resolve()
BASE_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = BASE_DIR / "tulu.jsonl"

# ---------- 2. 角色映射 ----------
ROLE_MAP = {
    "user":      "human",
    "human":     "human",
    "assistant": "gpt",
    "gpt":       "gpt",
}

# ---------- 3. 主處理程序 ----------
def process():
    print("🚀 啟動 Tulu 指令遵循資料集下載程序...")
    print("📦 資料來源: allenai/tulu-v2-sft-mixture")

    try:
        ds = load_dataset("allenai/tulu-v2-sft-mixture", split="train")
    except Exception as e:
        print(f"❌ 載入失敗：{e}")
        return

    valid_count = 0
    skipped_image = 0
    skipped_format = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
        for item in tqdm(ds, desc="轉換進度"):
            messages = item.get("messages", [])
            if not messages:
                skipped_format += 1
                continue

            # ---------- 跳過含圖片的條目（Stage 3 純文字防退化用途）----------
            has_image = False
            for msg in messages:
                content = msg.get("content", "")
                if isinstance(content, list):
                    # content 可能是 multimodal list
                    has_image = True
                    break
                if "<image>" in str(content):
                    has_image = True
                    break
            if has_image:
                skipped_image += 1
                continue

            # ---------- 轉換 messages -> conversations ----------
            conversations = []
            for msg in messages:
                role = ROLE_MAP.get(msg.get("role", "").lower())
                content = str(msg.get("content", "")).strip()
                if not role or not content:
                    continue
                conversations.append({"from": role, "value": content})

            # 最少需要一問一答
            if len(conversations) < 2:
                skipped_format += 1
                continue

            # 確保對話從 human 開始、gpt 結尾（奇偶對齊）
            if conversations[0]["from"] != "human":
                skipped_format += 1
                continue
            if conversations[-1]["from"] != "gpt":
                # 截掉尾部不完整的 human 發言
                conversations = conversations[:-1]
            if len(conversations) < 2:
                skipped_format += 1
                continue

            formatted_data = {
                "id": item.get("id", f"tulu_{valid_count}"),
                "conversations": conversations
            }

            f_out.write(json.dumps(formatted_data, ensure_ascii=False) + "\n")
            valid_count += 1

    print(f"\n✅ 處理完成！")
    print(f"  有效樣本: {valid_count} 筆")
    print(f"  跳過（含圖）: {skipped_image} 筆")
    print(f"  跳過（格式異常）: {skipped_format} 筆")
    print(f"📄 輸出檔案: {OUTPUT_FILE}")

    # 預覽
    try:
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            print("\n🔍 預覽第一筆:")
            print(f.readline())
    except Exception:
        pass

if __name__ == "__main__":
    process()
