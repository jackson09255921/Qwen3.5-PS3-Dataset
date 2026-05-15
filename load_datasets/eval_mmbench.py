import json
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset

BASE_DIR   = Path("/home/fireblue/datasets/eval/mmbench")
IMAGES_DIR = BASE_DIR / "images"
OUTPUT     = BASE_DIR / "test.jsonl"

BASE_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

LETTERS = ["A", "B", "C", "D"]


def _get_split(raw):
    keys = list(raw.keys())
    for prefer in ["dev", "test", "validation", "train"]:
        if prefer in keys:
            return raw[prefer]
    return raw[keys[0]]


def main():
    print("下載 opencompass/MMBench_DEV_EN ...")
    try:
        raw = load_dataset("opencompass/MMBench_DEV_EN")
    except Exception:
        print("  opencompass 失敗，改用 lmms-lab/MMBench ...")
        raw = load_dataset("lmms-lab/MMBench", "MMBench_DEV_EN")

    ds = _get_split(raw)
    print(f"  → {len(ds)} 筆，split 欄位: {ds.column_names}")

    records = []
    for i, ex in enumerate(tqdm(ds)):
        # 存圖
        img_name = f"{i:06d}.jpg"
        img_path = IMAGES_DIR / img_name
        if not img_path.exists():
            img = ex.get("image") or ex.get("img")
            if img is not None:
                img.convert("RGB").save(img_path, format="JPEG")

        # 選項：欄位可能是 A/B/C/D 或 option_A/option_B/...
        def _opt(letter):
            for key in [letter, f"option_{letter}", f"choice_{letter}"]:
                v = ex.get(key)
                if v is not None:
                    return str(v)
            return ""

        options = [f"{l}. {_opt(l)}" for l in LETTERS if _opt(l)]

        # 答案字母
        answer = str(ex.get("answer", ex.get("gt_answer", "A"))).strip().upper()
        if answer not in LETTERS:
            answer = LETTERS[0]

        # hint / context (可選)
        hint = ex.get("hint", ex.get("context", "")) or ""

        question = ex.get("question", ex.get("text", ""))
        if hint:
            question = f"{hint}\n{question}"

        records.append({
            "question_id": ex.get("index", ex.get("id", i)),
            "question":    question,
            "options":     options,
            "answer":      answer,
            "category":    ex.get("category", ex.get("l2-category", "")),
            "image":       str(img_path),
        })

    with open(OUTPUT, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n✅ {len(records)} 筆 → {OUTPUT}")


if __name__ == "__main__":
    main()
