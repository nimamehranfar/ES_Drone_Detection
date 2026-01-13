import argparse
import random
import shutil
from pathlib import Path


SRC_ROOT = Path(r"E:\Dataset\YOLOv11_mixed")
DST_ROOT_DEFAULT = Path(r"E:\Dataset\YOLOv11_subset_3k_801010")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(dir_path: Path) -> list[Path]:
    imgs = []
    for ext in IMG_EXTS:
        imgs.extend(dir_path.glob(f"*{ext}"))
    return sorted(imgs)


def ensure_dirs(dst_root: Path) -> None:
    for split in ("train", "val", "test"):
        (dst_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst_root / "labels" / split).mkdir(parents=True, exist_ok=True)


def copy_pair(img_src: Path, lbl_src: Path, img_dst: Path, lbl_dst: Path) -> None:
    shutil.copy2(img_src, img_dst)
    # label might be empty file, but must exist in your dataset layout
    shutil.copy2(lbl_src, lbl_dst)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--total", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dst", type=str, default=str(DST_ROOT_DEFAULT))
    args = ap.parse_args()

    total = args.total
    n_train = int(total * 0.8)
    n_val = int(total * 0.1)
    n_test = total - n_train - n_val

    src_train_img_dir = SRC_ROOT / "images" / "train"
    src_train_lbl_dir = SRC_ROOT / "labels" / "train"

    if not src_train_img_dir.exists():
        raise RuntimeError(f"Missing: {src_train_img_dir}")
    if not src_train_lbl_dir.exists():
        raise RuntimeError(f"Missing: {src_train_lbl_dir}")

    # Pool = ONLY from existing train split (cleanest; avoids leakage)
    pool_imgs = list_images(src_train_img_dir)

    # Keep only those with existing label file
    valid = []
    for img in pool_imgs:
        lbl = src_train_lbl_dir / f"{img.stem}.txt"
        if lbl.exists():
            valid.append((img, lbl))

    if len(valid) < total:
        raise RuntimeError(f"Not enough labeled samples in pool: {len(valid)} < {total}")

    random.seed(args.seed)
    sample = random.sample(valid, total)

    # Split
    train_set = sample[:n_train]
    val_set = sample[n_train:n_train + n_val]
    test_set = sample[n_train + n_val:]

    dst_root = Path(args.dst)
    ensure_dirs(dst_root)

    # Copy selected subset
    for split_name, items in (("train", train_set), ("val", val_set), ("test", test_set)):
        for img_src, lbl_src in items:
            img_dst = dst_root / "images" / split_name / img_src.name
            lbl_dst = dst_root / "labels" / split_name / lbl_src.name
            copy_pair(img_src, lbl_src, img_dst, lbl_dst)

    # Write split manifests (absolute paths, useful for debugging)
    for split_name in ("train", "val", "test"):
        manifest = dst_root / f"{split_name}.txt"
        lines = []
        for img in list_images(dst_root / "images" / split_name):
            lines.append(str(img))
        manifest.write_text("\n".join(lines), encoding="utf-8")

    print("DONE")
    print(f"Subset root: {dst_root}")
    print(f"train: {len(train_set)} | val: {len(val_set)} | test: {len(test_set)}")


if __name__ == "__main__":
    main()
