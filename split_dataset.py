import random
import shutil
from pathlib import Path

# 1. Auto-detect the folder where this script lives (big_batch)
# This fixes the VS Code working directory problem!
SCRIPT_DIR = Path(__file__).parent.resolve()

# Set SRC_DIR to the script folder (big_batch)
SRC_DIR = SCRIPT_DIR

# Set where the split dataset will be saved
OUT_DIR = SCRIPT_DIR / "split_dataset"

VAL_RATIO = 0.15  # 15% validation split (~164 pairs)
SEED = 42

random.seed(SEED)

# 2. Match images with their corresponding JSON files
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
pairs = {}

print(f"Searching for files in: {SRC_DIR}")

for file_path in SRC_DIR.glob("*"):
    stem = file_path.stem
    ext = file_path.suffix.lower()

    if ext in IMAGE_EXTS:
        pairs.setdefault(stem, {})["img"] = file_path
    elif ext == ".json":
        pairs.setdefault(stem, {})["json"] = file_path

# Filter out orphan images or orphan JSONs
valid_stems = [
    stem for stem, data in pairs.items() if "img" in data and "json" in data
]
print(f"Found {len(valid_stems)} valid image + JSON pairs.")

if len(valid_stems) == 0:
    print(
        "\n⚠️ Still 0 pairs? Check if your images/JSONs are inside a subfolder inside big_batch!"
    )
    exit()

# 3. Shuffle & Split
random.shuffle(valid_stems)
val_size = int(len(valid_stems) * VAL_RATIO)

val_stems = set(valid_stems[:val_size])
train_stems = set(valid_stems[val_size:])

# 4. Create directories and copy files
for split, stems in [("train", train_stems), ("val", val_stems)]:
    img_dir = OUT_DIR / split / "images"
    json_dir = OUT_DIR / split / "annotations"

    img_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    for stem in stems:
        shutil.copy(pairs[stem]["img"], img_dir / pairs[stem]["img"].name)
        shutil.copy(pairs[stem]["json"], json_dir / pairs[stem]["json"].name)

print(f"\nSuccessfully split dataset!")
print(f" - Train: {len(train_stems)} pairs -> {OUT_DIR / 'train'}")
print(f" - Val:   {len(val_stems)} pairs -> {OUT_DIR / 'val'}")