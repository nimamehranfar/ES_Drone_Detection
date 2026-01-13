import random
import shutil
from pathlib import Path
from tqdm import tqdm

# ==============================
# CONFIG (NO PLACEHOLDERS)
# ==============================

SRC_ROOT = Path(r"E:\Dataset\YOLOv11_ready_rgb")
OUT_ROOT = Path(r"E:\Dataset\YOLOv11_mixed")

SPLIT_RATIO = {
    "train": 0.8,
    "val": 0.1,
    "test": 0.1
}

RANDOM_SEED = 42

# ==============================
# DATASET SOURCES
# ==============================

DATASETS = {
    "anti_uav": SRC_ROOT / "anti_uav_rgbt_rgb",
    "dut": SRC_ROOT / "dut_anti_uav_det",
    "wosdetc": SRC_ROOT / "wosdetc_train",
}

# ==============================
# UTILS
# ==============================

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def collect_pairs(dataset_name: str, root: Path):
    pairs = []
    img_root = root / "images"
    lbl_root = root / "labels"

    for split_dir in img_root.rglob("*"):
        if not split_dir.is_dir():
            continue

        split_name = split_dir.name
        lbl_dir = lbl_root / split_name
        if not lbl_dir.exists():
            continue

        for img_path in split_dir.glob("*.jpg"):
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                continue
            pairs.append((dataset_name, img_path, lbl_path))

    return pairs

# ==============================
# MAIN
# ==============================

def main():
    random.seed(RANDOM_SEED)

    all_pairs = []

    for name, path in DATASETS.items():
        if not path.exists():
            raise RuntimeError(f"Dataset missing: {path}")
        pairs = collect_pairs(name, path)
        print(f"{name}: {len(pairs)} samples")
        all_pairs.extend(pairs)

    if not all_pairs:
        raise RuntimeError("No samples collected.")

    random.shuffle(all_pairs)

    total = len(all_pairs)
    n_train = int(total * SPLIT_RATIO["train"])
    n_val = int(total * SPLIT_RATIO["val"])

    splits = {
        "train": all_pairs[:n_train],
        "val": all_pairs[n_train:n_train + n_val],
        "test": all_pairs[n_train + n_val:]
    }

    # create output dirs
    for split in splits:
        ensure_dir(OUT_ROOT / "images" / split)
        ensure_dir(OUT_ROOT / "labels" / split)

    # copy files
    for split, items in splits.items():
        print(f"Writing {split}: {len(items)} samples")
        for dataset_name, img_src, lbl_src in tqdm(items, desc=split):
            new_stem = f"{dataset_name}_{img_src.stem}"

            img_dst = OUT_ROOT / "images" / split / f"{new_stem}.jpg"
            lbl_dst = OUT_ROOT / "labels" / split / f"{new_stem}.txt"

            shutil.copy2(img_src, img_dst)
            shutil.copy2(lbl_src, lbl_dst)

    # ==============================
    # CLEANUP
    # ==============================

    print("\nCleaning up intermediate datasets...")
    for path in DATASETS.values():
        shutil.rmtree(path)

    print("\nDONE.")
    print(f"Final dataset: {OUT_ROOT}")
    print(f"Total samples: {total}")
    for k, v in splits.items():
        print(f"{k}: {len(v)}")

if __name__ == "__main__":
    main()
