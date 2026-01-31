from pathlib import Path
import shutil

SRC_ROOT = Path(r"E:\Dataset\Anti-UAV.v2i.yolov11")      # Roboflow export root
DST_ROOT = Path(r"E:\Dataset\Anti-MUAV-roboflow.v2")        # YOLO-ready output
PREFIX = "anti-muav-roboflow_"

SPLIT_MAP = {
    "train": "train",
    "valid": "val",
    "test": "test",
}

for src_split, dst_split in SPLIT_MAP.items():
    src_img = SRC_ROOT / src_split / "images"
    src_lbl = SRC_ROOT / src_split / "labels"

    dst_img = DST_ROOT / "images" / dst_split
    dst_lbl = DST_ROOT / "labels" / dst_split

    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    for p in src_img.glob("*"):
        shutil.copy2(
            p,
            dst_img / f"{PREFIX}{p.name}"
        )

    for p in src_lbl.glob("*.txt"):
        shutil.copy2(
            p,
            dst_lbl / f"{PREFIX}{p.name}"
        )

print("DONE: YOLOv11 folder structure created")



import random

ROOT = Path(r"E:\Dataset\Anti-MUAV-roboflow.v2")
IMG = ROOT / "images"
LBL = ROOT / "labels"

EXTS = {".jpg", ".jpeg", ".png"}
RNG = random.Random(0)  # deterministic

def list_images(split: str):
    d = IMG / split
    if not d.exists():
        return []
    return [p for p in d.iterdir() if p.is_file() and p.suffix.lower() in EXTS]

def move_pair(img_path: Path, src_split: str, dst_split: str):
    # move image
    (IMG / dst_split).mkdir(parents=True, exist_ok=True)
    (LBL / dst_split).mkdir(parents=True, exist_ok=True)

    dst_img = IMG / dst_split / img_path.name
    shutil.move(str(img_path), str(dst_img))

    # move matching label if it exists
    src_lbl = LBL / src_split / f"{img_path.stem}.txt"
    if src_lbl.exists():
        dst_lbl = LBL / dst_split / src_lbl.name
        shutil.move(str(src_lbl), str(dst_lbl))

def counts():
    tr = len(list_images("train"))
    va = len(list_images("val"))
    te = len(list_images("test"))
    return tr, va, te, tr + va + te

# --- targets ---
tr0, va0, te0, total = counts()
target_train = int(total * 0.80)
target_val   = int(total * 0.10)
target_test  = total - target_train - target_val  # exact remainder

print(f"Initial: train={tr0}, val={va0}, test={te0}, total={total}")
print(f"Target : train={target_train}, val={target_val}, test={target_test}")

# --- Step A: grow train to target by pulling from val/test ---
train_imgs = list_images("train")
val_imgs = list_images("val")
test_imgs = list_images("test")

pool = [("val", p) for p in val_imgs] + [("test", p) for p in test_imgs]
RNG.shuffle(pool)

while len(train_imgs) < target_train and pool:
    src, img = pool.pop()
    move_pair(img, src, "train")
    train_imgs.append(img)

# --- Step B: rebalance val/test to exact targets by moving between them ---
# refresh lists after moves
val_imgs = list_images("val")
test_imgs = list_images("test")

# If val too big, move val -> test
if len(val_imgs) > target_val:
    extra = len(val_imgs) - target_val
    RNG.shuffle(val_imgs)
    for img in val_imgs[:extra]:
        move_pair(img, "val", "test")

# If test too big, move test -> val
val_imgs = list_images("val")
test_imgs = list_images("test")

if len(test_imgs) > target_test:
    extra = len(test_imgs) - target_test
    RNG.shuffle(test_imgs)
    for img in test_imgs[:extra]:
        move_pair(img, "test", "val")

# Final report
tr, va, te, total2 = counts()
print(f"Final  : train={tr}, val={va}, test={te}, total={total2}")