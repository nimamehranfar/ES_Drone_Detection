import argparse
from collections import defaultdict
from pathlib import Path

import cv2
from tqdm import tqdm


IN_ROOT = Path(r"E:\Dataset\wosdetc_train_videos")
ANN_DIR = IN_ROOT / "challenge-master" / "annotations"

OUT_ROOT = Path(r"E:\Dataset\YOLOv11_ready_rgb\wosdetc_train")
OUT_IMG_DIR = OUT_ROOT / "images" / "train"
OUT_LBL_DIR = OUT_ROOT / "labels" / "train"


VIDEO_EXTS = {".mp4", ".avi", ".mpg"}


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def index_videos(root: Path) -> dict[str, Path]:
    # map stem -> video path (first match)
    vid_map: dict[str, Path] = {}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            # skip extracted images area if any
            stem = p.stem
            if stem not in vid_map:
                vid_map[stem] = p
    return vid_map


def parse_annotation_txt(txt_path: Path):
    """
    Handles BOTH WOSDETC formats:
    - one object per line
    - multiple objects per line (swarm)

    Returns:
        dict[int, list[tuple[str, int, int, int, int]]]
        frame_id -> list of (label, x, y, w, h)
    """
    from collections import defaultdict

    per_frame = defaultdict(list)

    for line in txt_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 7:
            continue

        frame_id = int(parts[0])
        # track_id = parts[1]  # ignored on purpose

        obj_parts = parts[2:]

        # obj_parts must be multiple of 5: x y w h label
        if len(obj_parts) % 5 != 0:
            raise ValueError(
                f"Malformed annotation line in {txt_path}:\n{line}"
            )

        for i in range(0, len(obj_parts), 5):
            x = int(float(obj_parts[i]))
            y = int(float(obj_parts[i + 1]))
            w = int(float(obj_parts[i + 2]))
            h = int(float(obj_parts[i + 3]))
            label = obj_parts[i + 4].lower()

            if w <= 0 or h <= 0:
                continue

            per_frame[frame_id].append((label, x, y, w, h))

    return per_frame



def write_yolo(label_path: Path, boxes: list[tuple[str, int, int, int, int]], img_w: int, img_h: int, drone_only: bool) -> None:
    # class map
    # drone -> 0, bird -> 1 (kept for later training)
    lines = []
    for label, x, y, w, h in boxes:
        if drone_only and label != "drone":
            continue

        if label == "drone":
            cls = 0
        elif label == "bird":
            cls = 1
        else:
            # unknown label: skip
            continue

        if w <= 0 or h <= 0:
            continue

        cx = (x + w / 2.0) / img_w
        cy = (y + h / 2.0) / img_h
        nw = w / img_w
        nh = h / img_h

        cx = min(max(cx, 0.0), 1.0)
        cy = min(max(cy, 0.0), 1.0)
        nw = min(max(nw, 0.0), 1.0)
        nh = min(max(nh, 0.0), 1.0)

        lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

    label_path.write_text("\n".join(lines), encoding="utf-8")


def process_one_session(txt_path: Path, video_path: Path, every_n: int, drone_only: bool) -> None:
    per_frame = parse_annotation_txt(txt_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return

    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    session = txt_path.stem

    ensure_dir(OUT_IMG_DIR)
    ensure_dir(OUT_LBL_DIR)

    pbar = tqdm(total=total, desc=f"WOSDETC {session}", unit="frame")
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if idx % every_n == 0:
            out_stem = f"{session}_{idx:06d}"
            out_img = OUT_IMG_DIR / f"{out_stem}.jpg"
            out_lbl = OUT_LBL_DIR / f"{out_stem}.txt"

            cv2.imwrite(str(out_img), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

            boxes = per_frame.get(idx, [])
            # empty labels allowed (negative frame)
            write_yolo(out_lbl, boxes, img_w, img_h, drone_only=drone_only)

        idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--every-n", type=int, default=2, help="Save 1 frame every N frames (default: 5)")
    ap.add_argument("--drone-only", action="store_true", help="If set, ignore 'bird' annotations and keep only drones")
    args = ap.parse_args()

    vid_map = index_videos(IN_ROOT)

    txt_files = sorted(ANN_DIR.glob("*.txt"))
    for txt_path in txt_files:
        session = txt_path.stem
        video_path = vid_map.get(session)
        if video_path is None:
            # no matching video found; skip
            continue
        process_one_session(txt_path, video_path, args.every_n, args.drone_only)

    print(f"\nDONE. Output: {OUT_ROOT}")


if __name__ == "__main__":
    main()
