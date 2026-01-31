import os
import cv2
import argparse
import random
from pathlib import Path

FRAME_STRIDE = 3          # extract 1 in 4 frames
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

random.seed(42)

def extract_frames(video_dir):
    frames = []

    for video_name in os.listdir(video_dir):
        if not video_name.endswith((".mp4", ".avi")):
            continue

        video_path = video_dir / video_name
        cap = cv2.VideoCapture(str(video_path))

        frame_idx = 0
        saved_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % FRAME_STRIDE == 0:
                frame_name = f"{video_name.split('.')[0]}_{saved_idx:06d}.jpg"
                frames.append((frame, frame_name))
                saved_idx += 1

            frame_idx += 1

        cap.release()

    return frames

def write_split(frames, img_dir, lbl_dir):
    for frame, name in frames:
        img_path = img_dir / name
        lbl_path = lbl_dir / name.replace(".jpg", ".txt")

        cv2.imwrite(str(img_path), frame)
        lbl_path.write_text("")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_root_path',
        default=r"E:\Dataset_zip\FBD-SV-2024",
        type=str,
        help='Path to FBD-SV-2024 root'
    )
    args = parser.parse_args()

    root = Path(args.data_root_path)

    video_dirs = [
        root / "videos" / "train",
        root / "videos" / "val"
    ]

    all_frames = []

    for vdir in video_dirs:
        all_frames.extend(extract_frames(vdir))

    if len(all_frames) == 0:
        raise RuntimeError("No frames extracted")

    random.shuffle(all_frames)

    total = len(all_frames)
    n_train = int(total * TRAIN_RATIO)
    n_val = int(total * VAL_RATIO)

    train_frames = all_frames[:n_train]
    val_frames = all_frames[n_train:n_train + n_val]
    test_frames = all_frames[n_train + n_val:]

    yolo_root = root / "yolo26"

    for split in ["train", "val", "test"]:
        (yolo_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (yolo_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    write_split(
        train_frames,
        yolo_root / "images" / "train",
        yolo_root / "labels" / "train"
    )

    write_split(
        val_frames,
        yolo_root / "images" / "val",
        yolo_root / "labels" / "val"
    )

    write_split(
        test_frames,
        yolo_root / "images" / "test",
        yolo_root / "labels" / "test"
    )

    print(f"Total frames extracted: {total}")
    print(f"Train: {len(train_frames)} | Val: {len(val_frames)} | Test: {len(test_frames)}")
    print("YOLOv26 dataset with empty annotations created.")
