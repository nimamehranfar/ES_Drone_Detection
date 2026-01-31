import random
import shutil
from pathlib import Path
from tqdm import tqdm
import argparse

# ==============================
# CONFIG (DEFAULTS)
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
# DATASET SOURCES (DEFAULTS)
# ==============================

DATASETS = {
    # "anti_uav": SRC_ROOT / "anti_uav_rgbt_rgb",
    # "dut": SRC_ROOT / "dut_anti_uav_det",
    "wosdetc": SRC_ROOT / "wosdetc_train",
}

# ==============================
# UTILS
# ==============================

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def collect_pairs(dataset_name: str, root: Path):
    """
    Supports TWO layouts:
    1) images/<split>/*.jpg + labels/<split>/*.txt
    2) Images/*.jpg + Labels/*.txt  (flat mixed dataset)
    """
    pairs = []

    # ---- mixed flat mode ----
    # img_root_flat = root / "Images"
    # lbl_root_flat = root / "Labels"
    #
    # if img_root_flat.exists() and lbl_root_flat.exists():
    #     for img_path in img_root_flat.glob("*.jpg"):
    #         lbl_path = lbl_root_flat / f"{img_path.stem}.txt"
    #         if lbl_path.exists():
    #             pairs.append((dataset_name, img_path, lbl_path))
    #     return pairs


    # ---- split-based mode (original behavior) ----
    img_root = root / "images"
    lbl_root = root / "labels"

    for split_dir in img_root.iterdir():
        if not split_dir.is_dir():
            continue

        split_name = split_dir.name
        lbl_dir = lbl_root / split_name
        if not lbl_dir.exists():
            continue

        for img_path in split_dir.glob("*.jpg"):
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if lbl_path.exists():
                pairs.append((dataset_name, img_path, lbl_path))

    return pairs

# ==============================
# MAIN
# ==============================

def main():
    global SRC_ROOT, OUT_ROOT, SPLIT_RATIO, RANDOM_SEED, DATASETS

    parser = argparse.ArgumentParser()
    parser.add_argument("--src-root", type=Path)
    parser.add_argument("--out-root", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--train", type=float)
    parser.add_argument("--val", type=float)
    parser.add_argument("--test", type=float)

    args = parser.parse_args()

    if args.src_root:
        SRC_ROOT = args.src_root

    if args.out_root:
        OUT_ROOT = args.out_root

    if args.seed is not None:
        RANDOM_SEED = args.seed

    if args.train or args.val or args.test:
        SPLIT_RATIO = {
            "train": args.train or SPLIT_RATIO["train"],
            "val":   args.val   or SPLIT_RATIO["val"],
            "test":  args.test  or SPLIT_RATIO["test"],
        }

    if args.src_root:
        DATASETS = {"mixed": SRC_ROOT}
    else:
        DATASETS = {
            # "anti_uav": SRC_ROOT / "anti_uav_rgbt_rgb",
            # "dut": SRC_ROOT / "dut_anti_uav_det",
            "wosdetc": SRC_ROOT / "wosdetc_train",
        }

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

    for split in splits:
        ensure_dir(OUT_ROOT / "images" / split)
        ensure_dir(OUT_ROOT / "labels" / split)

    for split, items in splits.items():
        print(f"Writing {split}: {len(items)} samples")
        for dataset_name, img_src, lbl_src in tqdm(items, desc=split):
            new_stem = img_src.stem if dataset_name == "mixed" else f"{dataset_name}_{img_src.stem}"

            shutil.copy2(
                img_src,
                OUT_ROOT / "images" / split / f"{new_stem}.jpg"
            )
            shutil.copy2(
                lbl_src,
                OUT_ROOT / "labels" / split / f"{new_stem}.txt"
            )

    print("\nDONE.")
    print(f"Final dataset: {OUT_ROOT}")
    print(f"Total samples: {total}")
    for k, v in splits.items():
        print(f"{k}: {len(v)}")

if __name__ == "__main__":
    main()
