import json
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset

BASE_DIR   = Path("/home/fireblue/datasets/eval/vstar")
IMAGES_DIR = BASE_DIR / "images"
OUTPUT     = BASE_DIR / "test.jsonl"

BASE_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

LETTERS = ["A", "B", "C", "D"]


def _get_split(raw):
    keys = list(raw.keys())
    for prefer in ["test", "validation", "train"]:
        if prefer in keys:
            return raw[prefer]
    return raw[keys[0]]


def main():
    print("下載 craigwu/vstar_bench ...")
    raw = load_dataset("craigwu/vstar_bench")
    ds = _get_split(raw)
    print(f"  → {len(ds)} 筆，欄位: {ds.column_names}")

    # 找 subset 欄位名稱
    subset_field = next(
        (f for f in ["subset", "category", "type", "split_type"] if f in ds.column_names),
        None,
    )

    records = []
    for i, ex in enumerate(tqdm(ds)):
        subset = ex.get(subset_field, "default") if subset_field else "default"
        img_name = f"{i:06d}.jpg"
        img_path = IMAGES_DIR / img_name
        if not img_path.exists():
            ex["image"].convert("RGB").save(img_path, format="JPEG")

            # answer 可能是 int index (0-3) 或字母 "A"-"D"
            raw_ans = ex.get("answer", ex.get("label", 0))
            if isinstance(raw_ans, int):
                answer_letter = LETTERS[raw_ans]
            elif isinstance(raw_ans, str) and raw_ans in LETTERS:
                answer_letter = raw_ans
            else:
                # 嘗試從 options 列表中匹配
                options = ex.get("options", [])
                try:
                    answer_letter = LETTERS[options.index(raw_ans)]
                except (ValueError, IndexError):
                    answer_letter = str(raw_ans)

            options = ex.get("options", [])
            # 確保 options 帶字母前綴
            fmt_options = []
            for j, opt in enumerate(options):
                prefix = f"{LETTERS[j]}. "
                fmt_options.append(opt if opt.startswith(prefix) else prefix + opt)

            records.append({
                "question_id": f"{subset}_{i}",
                "subset":      subset,
                "question":    ex["question"],
                "options":     fmt_options,
                "answer":      answer_letter,
                "image":       str(img_path),
            })

    with open(OUTPUT, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total = len(records)
    by_subset = {s: sum(1 for r in records if r["subset"] == s) for s in SUBSETS}
    print(f"\n✅ {total} 筆 → {OUTPUT}")
    for s, n in by_subset.items():
        print(f"   {s}: {n}")


if __name__ == "__main__":
    main()
