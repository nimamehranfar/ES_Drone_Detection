from pathlib import Path

# ================= CONFIG =================

YOLO_ROOT = Path(r"E:\Dataset\Suppress False Positive\FBD-SV")  # change per dataset
PREFIX = YOLO_ROOT.name + "_"

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

# ================= RENAME =================

for split in ["train", "val", "test"]:
    img_dir = YOLO_ROOT / "images" / split
    lbl_dir = YOLO_ROOT / "labels" / split

    if not img_dir.exists() or not lbl_dir.exists():
        continue

    for img_path in img_dir.iterdir():
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue

        if img_path.name.startswith(PREFIX):
            continue  # already renamed

        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue

        new_img_name = PREFIX + img_path.name
        new_lbl_name = PREFIX + lbl_path.name

        img_path.rename(img_dir / new_img_name)
        lbl_path.rename(lbl_dir / new_lbl_name)

print(f"Renaming complete for dataset: {YOLO_ROOT.name}")
