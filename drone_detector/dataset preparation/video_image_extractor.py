import argparse
import random
import shutil
import hashlib
from pathlib import Path
from bisect import bisect_right

import cv2

def short_hash(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:8]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Folder containing VIRAT .mp4 files (searched recursively)")
    ap.add_argument("--out", required=True, help="Output root that will contain Images/ and Labels/")
    ap.add_argument("--dataset", default="VIRAT", help="Prefix for output filenames")
    ap.add_argument("--n", type=int, required=True, help="How many frames to sample in total")
    ap.add_argument("--every", type=int, default=5, help="Take 1 frame every N frames")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    out_images = out / "Images"
    out_labels = out / "Labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    videos = sorted([p for p in src.rglob("*.mp4") if p.is_file()])
    if not videos:
        raise SystemExit(f"No .mp4 found under: {src}")

    # Pass 1: compute eligible frame counts per video
    elig_counts = []
    for v in videos:
        cap = cv2.VideoCapture(str(v))
        if not cap.isOpened():
            cap.release()
            elig_counts.append(0)
            continue
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        # eligible frames are indices: 0, every, 2*every, ...
        elig = (total_frames + (args.every - 1)) // args.every
        elig_counts.append(max(0, elig))

    total_elig = sum(elig_counts)
    if total_elig == 0:
        raise SystemExit("No eligible frames found (frame_count read failed?)")

    n = min(args.n, total_elig)
    rng = random.Random(args.seed)

    # Sample n unique global indices in [0, total_elig)
    picks_global = sorted(rng.sample(range(total_elig), n))

    # Build prefix sums for mapping global index -> (video_idx, local_idx)
    prefix = []
    s = 0
    for c in elig_counts:
        s += c
        prefix.append(s)

    # Group requested frame indices per video
    per_video = {i: [] for i in range(len(videos))}
    for g in picks_global:
        vidx = bisect_right(prefix, g)
        prev = 0 if vidx == 0 else prefix[vidx - 1]
        local = g - prev  # local eligible index
        frame_index = local * args.every
        per_video[vidx].append(frame_index)

    saved = 0
    for vidx, frame_indices in per_video.items():
        if not frame_indices:
            continue

        vpath = videos[vidx]
        cap = cv2.VideoCapture(str(vpath))
        if not cap.isOpened():
            cap.release()
            continue

        frame_indices = sorted(set(frame_indices))
        for fi in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if not ret:
                continue

            h = short_hash(f"{vpath}|{fi}")
            out_name = f"{args.dataset}_{saved:06d}_{vpath.stem}_{fi:08d}_{h}.jpg"
            img_path = out_images / out_name
            lbl_path = out_labels / (Path(out_name).stem + ".txt")

            cv2.imwrite(str(img_path), frame)
            lbl_path.write_text("", encoding="utf-8")  # empty label = no objects
            saved += 1

        cap.release()

    print(f"Done: {saved} VIRAT frames sampled -> {out_images}")
    print(f"YOLO empty labels created -> {out_labels}")

if __name__ == "__main__":
    main()
