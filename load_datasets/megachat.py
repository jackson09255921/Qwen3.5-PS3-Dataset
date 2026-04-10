import json
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset

# ---------- 1. 配置 ----------
BASE_DIR = Path("../megachat_gpt4o").resolve()
BASE_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = BASE_DIR / "megachat_train.jsonl"

# ---------- 2. Task-oriented 過濾器 ----------
KEEP_KEYWORDS = [
    "how do i", "how to", "step by step", "help me",
    "guide me", "set up", "install", "open ",
    "configure", "use ", "create a new", "click",
    "navigate", "settings",
]
SKIP_KEYWORDS = ["summarize", "summary of", "explain", "what are the main ideas"]

def is_task_oriented(conversations):
    first_user = next((m for m in conversations if m.get("role") == "user"), None)
    if not first_user:
        return False
    q = first_user["content"].lower()
    if any(b in q for b in SKIP_KEYWORDS):
        return False
    return any(k in q for k in KEEP_KEYWORDS)

# ---------- 3. 角色映射 ----------
ROLE_MAP = {"user": "human", "assistant": "gpt"}

# ---------- 4. 主程序 ----------
def main():
    print("🚀 啟動 MegaChat 資料集處理程序...")
    print("📦 資料來源: xiaozheyao/megachat (sharegpt)")

    try:
        ds = load_dataset("xiaozheyao/megachat", name="sharegpt", split="train")
    except Exception as e:
        print(f"❌ 載入失敗：{e}")
        return

    kept = 0
    skipped_filter = 0
    skipped_format = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
        for item in tqdm(ds, desc="過濾與轉換"):
            conv = item.get("conversations", [])
            if not conv:
                skipped_format += 1
                continue

            # Task-oriented 過濾
            if not is_task_oriented(conv):
                skipped_filter += 1
                continue

            # 轉換 role -> from
            conversations = []
            for msg in conv:
                role = ROLE_MAP.get(msg.get("role", "").lower())
                content = str(msg.get("content", "")).strip()
                if not role or not content:
                    continue
                conversations.append({"from": role, "value": content})

            # 確保從 human 開始、gpt 結尾
            if len(conversations) < 2 or conversations[0]["from"] != "human":
                skipped_format += 1
                continue
            if conversations[-1]["from"] != "gpt":
                conversations = conversations[:-1]
            if len(conversations) < 2:
                skipped_format += 1
                continue

            formatted = {
                "id": f"megachat_{kept:06d}",
                "conversations": conversations,
                "meta": {
                    "src_dataset": "megachat_gpt4o",
                    "source": item.get("meta", {}).get("source") if isinstance(item.get("meta"), dict) else None,
                }
            }

            f_out.write(json.dumps(formatted, ensure_ascii=False) + "\n")
            kept += 1

    print(f"\n✅ 處理完成！")
    print(f"  保留樣本: {kept} 筆")
    print(f"  跳過（非 task-oriented）: {skipped_filter} 筆")
    print(f"  跳過（格式異常）: {skipped_format} 筆")
    print(f"📄 輸出: {OUTPUT_FILE}")

    # 預覽
    try:
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            print("\n🔍 預覽第一筆:")
            print(f.readline())
    except Exception:
        pass

if __name__ == "__main__":
    main()
