import argparse
import hashlib
import random
import shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def short_hash(p: Path) -> str:
    return hashlib.md5(str(p).encode("utf-8")).hexdigest()[:8]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Source root to search recursively for images")
    ap.add_argument("--out", required=True, help="Output root that will contain Images/ and Labels/")
    ap.add_argument("--dataset", required=True, help="Dataset name prefix (e.g., BDD100K, UADETRAC)")
    ap.add_argument("--n", type=int, required=True, help="How many images to sample")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    out_images = out / "Images"
    out_labels = out / "Labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    all_imgs = [p for p in src.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    if not all_imgs:
        raise SystemExit(f"No images found under: {src}")

    rng = random.Random(args.seed)
    n = min(args.n, len(all_imgs))
    picks = rng.sample(all_imgs, n)

    for i, p in enumerate(picks):
        h = short_hash(p)
        safe_stem = p.stem.replace(" ", "_")
        out_name = f"{args.dataset}_{i:06d}_{safe_stem}_{h}{p.suffix.lower()}"
        dst_img = out_images / out_name
        dst_lbl = out_labels / (Path(out_name).stem + ".txt")

        shutil.copy2(p, dst_img)
        dst_lbl.write_text("", encoding="utf-8")  # empty label = no objects

    print(f"Done: {n} images copied from {src} -> {out_images}")
    print(f"YOLO empty labels created -> {out_labels}")

if __name__ == "__main__":
    main()
