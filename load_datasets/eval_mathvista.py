import json
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset

# 對齊 default.yaml: mathvista.testmini_data_path / test_data_path
BASE_DIR   = Path("/home/fireblue/datasets/eval/mathvista")
IMAGES_DIR = BASE_DIR / "images"
BASE_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = {
    "testmini": BASE_DIR / "mathvista_testmini.jsonl",
    "test":     BASE_DIR / "mathvista_test.jsonl",
}

def main():
    print("🚀 下載 MathVista (AI4Math/MathVista)...")
    ds_all = load_dataset("AI4Math/MathVista")

    for split, output_file in SPLITS.items():
        if split not in ds_all:
            print(f"  ⚠️  split '{split}' 不存在，跳過")
            continue

        ds = ds_all[split]
        print(f"\n🎯 {split} ({len(ds)} 筆)")

        with open(output_file, "w", encoding="utf-8") as f:
            for ex in tqdm(ds, desc=split):
                # 保留原始相對路徑結構 images/N.jpg，存到 BASE_DIR/images/N.jpg
                rel_path = ex["image"]                  # e.g. "images/1.jpg"
                abs_path = BASE_DIR / rel_path
                abs_path.parent.mkdir(parents=True, exist_ok=True)

                if not abs_path.exists() and ex.get("decoded_image") is not None:
                    ex["decoded_image"].convert("RGB").save(abs_path)

                # eval script: problem["image"] → Image.open(img_path)
                rec = {k: v for k, v in ex.items() if k != "decoded_image"}
                rec["image"] = str(abs_path)  # 絕對路徑，Image.open 可直接讀

                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        count = sum(1 for _ in open(output_file))
        print(f"  ✅ {count} 筆 → {output_file.name}")

    print(f"\n🎉 完成！")

if __name__ == "__main__":
    main()
