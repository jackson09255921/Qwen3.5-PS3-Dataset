from pathlib import Path
from PIL import Image

path = Path("coyo-700m-1M/images")
# path = Path("LLaVA-CC3M-Pretrain-595K/images")
count = 0  # Initialize a counter

print("--- Processing the first 100 images ---")

for file in path.rglob("*.jpg"):
    # 1. Stop the loop once 100 images have been processed
    if count >= 100:
        break

    try:
        # 2. Open the image and get dimensions
        img = Image.open(file)
        w, h = img.size
        print(f"[{count + 1:3d}] {file}: width={w}, height={h}")

        # 3. Increment the counter
        count += 1

    except Exception as e:
        # Handle cases where the file might be corrupted or not a valid image
        print(f"Error opening or reading image {file}: {e}")

print("--- Loop finished (Processed 100 items or reached end of files) ---")