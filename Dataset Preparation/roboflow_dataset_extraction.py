from pathlib import Path
import shutil
import random

# ============================
# CONFIG
# ============================

SRC_ROOT = Path(r"E:\Dataset\Suppress False Positive\Insect_videos.v14i.yolo26")   # Roboflow dataset root
DST_ROOT = Path(r"E:\Dataset\Suppress False Positive\Insect_videos.v14i.yolo26")      # Final YOLO dataset
PREFIX = "insect_roboflow_"                                 # optional, can be ""

SPLITS_IN = ["train", "valid", "test"]
SPLITS_OUT = ["train", "val", "test"]

RATIO = {
    "train": 0.8,
    "val": 0.1,
    "test": 0.1,
}

EXTS = {".jpg", ".jpeg", ".png"}
RNG = random.Random(42)

# ============================
# PREP OUTPUT DIRS
# ============================

for s in SPLITS_OUT:
    (DST_ROOT / "images" / s).mkdir(parents=True, exist_ok=True)
    (DST_ROOT / "labels" / s).mkdir(parents=True, exist_ok=True)

# ============================
# COLLECT ALL IMAGE/LABEL PAIRS
# ============================

pairs = []

for split in SPLITS_IN:
    img_dir = SRC_ROOT / split / "images"
    lbl_dir = SRC_ROOT / split / "labels"

    if not img_dir.exists():
        continue

    for img in img_dir.iterdir():
        if img.suffix.lower() not in EXTS:
            continue

        lbl = lbl_dir / f"{img.stem}.txt"
        if not lbl.exists():
            continue

        pairs.append((img, lbl))

if not pairs:
    raise RuntimeError("No image/label pairs found")

print(f"Collected {len(pairs)} samples")

# ============================
# SHUFFLE + SPLIT
# ============================

RNG.shuffle(pairs)
total = len(pairs)

n_train = int(total * RATIO["train"])
n_val   = int(total * RATIO["val"])

splits = {
    "train": pairs[:n_train],
    "val":   pairs[n_train:n_train + n_val],
    "test":  pairs[n_train + n_val:],
}

# ============================
# COPY TO YOLO26 STRUCTURE
# ============================

for split, items in splits.items():
    print(f"Writing {split}: {len(items)}")

    for img, lbl in items:
        new_name = PREFIX + img.name

        shutil.copy2(
            img,
            DST_ROOT / "images" / split / new_name
        )

        shutil.copy2(
            lbl,
            DST_ROOT / "labels" / split / f"{Path(new_name).stem}.txt"
        )

# ============================
# FINAL REPORT
# ============================

print("\nDONE")
for s in SPLITS_OUT:
    n = len(list((DST_ROOT / "images" / s).iterdir()))
    print(f"{s}: {n}")
