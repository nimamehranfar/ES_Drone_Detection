import argparse
import json
import os
import shutil
from pathlib import Path

import cv2
from tqdm import tqdm


IN_ROOT = Path(r"E:\Dataset\Anti-UAV-RGBT")  # RGB dataset root (NOT infrared)
OUT_ROOT = Path(r"E:\Dataset\YOLOv11_ready_rgb\anti_uav_rgbt_rgb")

SPLITS = ["train", "val", "test"]


def find_visible_json_and_video(seq_dir: Path) -> tuple[Path, Path] | tuple[None, None]:
    # visible.json required; mp4 required
    vjson = seq_dir / "visible.json"
    if not vjson.exists():
        # sometimes naming differs; fallback: any json with 'visible' in name
        cands = list(seq_dir.glob("*visible*.json"))
        vjson = cands[0] if cands else None

    videos = list(seq_dir.glob("*.mp4")) + list(seq_dir.glob("*.avi")) + list(seq_dir.glob("*.mpg"))
    video = videos[0] if videos else None

    if vjson is None or video is None:
        return None, None
    return vjson, video


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def write_yolo_label(label_path: Path, boxes_xywh: list[list[float]], img_w: int, img_h: int) -> None:
    # class: 0 (drone)
    # boxes in [x,y,w,h] pixels
    lines = []
    for x, y, w, h in boxes_xywh:
        cx = (x + w / 2.0) / img_w
        cy = (y + h / 2.0) / img_h
        nw = w / img_w
        nh = h / img_h
        # clamp for safety
        cx = min(max(cx, 0.0), 1.0)
        cy = min(max(cy, 0.0), 1.0)
        nw = min(max(nw, 0.0), 1.0)
        nh = min(max(nh, 0.0), 1.0)
        lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

    label_path.write_text("\n".join(lines), encoding="utf-8")


def process_sequence(split: str, seq_dir: Path, every_n: int) -> None:
    vjson, video = find_visible_json_and_video(seq_dir)
    if vjson is None or video is None:
        return

    with vjson.open("r", encoding="utf-8") as f:
        ann = json.load(f)

    # Expected format from your sample: {"exist":[...], "gt_rect":[[x,y,w,h], ...]}
    exist = ann.get("exist", [])
    gt_rect = ann.get("gt_rect", [])

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return

    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    max_frames = min(frame_count, len(exist), len(gt_rect))

    out_img_dir = OUT_ROOT / "images" / split
    out_lbl_dir = OUT_ROOT / "labels" / split
    ensure_dir(out_img_dir)
    ensure_dir(out_lbl_dir)

    seq_name = seq_dir.name

    frame_idx = 0
    saved = 0

    pbar = tqdm(total=max_frames, desc=f"Anti-UAV-RGBT RGB {split}/{seq_name}", unit="frame")
    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % every_n == 0:
            out_stem = f"{seq_name}_{frame_idx:06d}"
            out_img = out_img_dir / f"{out_stem}.jpg"
            out_lbl = out_lbl_dir / f"{out_stem}.txt"

            cv2.imwrite(str(out_img), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

            # write label (empty file allowed for negatives)
            if int(exist[frame_idx]) == 1:
                box = gt_rect[frame_idx]
                # Some frames can have [0,0,0,0] even when exist=1; filter it
                if isinstance(box, list) and len(box) == 4 and (box[2] > 0 and box[3] > 0):
                    write_yolo_label(out_lbl, [box], img_w, img_h)
                else:
                    out_lbl.write_text("", encoding="utf-8")
            else:
                out_lbl.write_text("", encoding="utf-8")

            saved += 1

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--every-n", type=int, default=5, help="Save 1 frame every N frames (default: 5)")
    args = ap.parse_args()

    for split in SPLITS:
        split_dir = IN_ROOT / split
        if not split_dir.exists():
            continue
        # each child is a sequence folder containing mp4 + visible.json
        for seq_dir in sorted([p for p in split_dir.iterdir() if p.is_dir()]):
            process_sequence(split, seq_dir, args.every_n)

    print(f"\nDONE. Output: {OUT_ROOT}")


if __name__ == "__main__":
    main()
