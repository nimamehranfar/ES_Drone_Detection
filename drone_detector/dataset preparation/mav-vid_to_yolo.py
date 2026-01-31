import random
import shutil
from pathlib import Path

# ================= CONFIG =================

SOURCE_ROOT = Path(r"E:\Dataset\mav_vid_dataset")
DEST_ROOT = Path(r"E:\Dataset\mav_vid")

TOTAL_IMAGES = 10_000
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

random.seed(42)

# ================= SETUP =================

for split in ["train", "val", "test"]:
    (DEST_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
    (DEST_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

# ================= COLLECT VALID PAIRS =================

valid_pairs = []

for split in ["train", "val"]:
    img_dir = SOURCE_ROOT / split / "img"

    if not img_dir.exists():
        continue

    for img_path in img_dir.iterdir():
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue

        label_path = img_path.with_suffix(".txt")

        if not label_path.exists():
            continue

        if label_path.stat().st_size == 0:
            continue  # must contain at least one annotation

        valid_pairs.append((img_path, label_path))

print(f"Found {len(valid_pairs)} annotated images")

if len(valid_pairs) < TOTAL_IMAGES:
    raise RuntimeError("Not enough annotated images in MAV-VID")

# ================= SAMPLE =================

selected = random.sample(valid_pairs, TOTAL_IMAGES)
random.shuffle(selected)

n_train = int(TOTAL_IMAGES * TRAIN_RATIO)
n_val = int(TOTAL_IMAGES * VAL_RATIO)

train_set = selected[:n_train]
val_set = selected[n_train:n_train + n_val]
test_set = selected[n_train + n_val:]

splits = {
    "train": train_set,
    "val": val_set,
    "test": test_set,
}

# ================= COPY FILES =================

def copy_split(pairs, split):
    for img_src, lbl_src in pairs:
        shutil.copy2(
            img_src,
            DEST_ROOT / "images" / split / img_src.name
        )
        shutil.copy2(
            lbl_src,
            DEST_ROOT / "labels" / split / lbl_src.name
        )

for split, pairs in splits.items():
    copy_split(pairs, split)
    print(f"{split}: {len(pairs)} images")

print("MAV-VID YOLOv26 dataset creation complete.")
