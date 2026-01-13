import shutil
from pathlib import Path

SRC_ROOT = Path(r"E:\Dataset\Anti-UAV-RGBT")
DST_ROOT = Path(r"E:\Dataset\Anti-UAV-RGBT-separated")

MODALITY_MAP = {
    "visible": "rgb",
    "infrared": "infrared"
}

for split in ["train", "val", "test"]:
    split_dir = SRC_ROOT / split
    if not split_dir.exists():
        continue

    for seq_dir in split_dir.iterdir():
        if not seq_dir.is_dir():
            continue

        for stem, modality in MODALITY_MAP.items():
            video = seq_dir / f"{stem}.mp4"
            ann   = seq_dir / f"{stem}.json"

            if not video.exists() or not ann.exists():
                continue

            dst_seq = DST_ROOT / modality / split / seq_dir.name
            dst_seq.mkdir(parents=True, exist_ok=True)

            shutil.copy2(video, dst_seq / video.name)
            shutil.copy2(ann, dst_seq / ann.name)

            print(f"[{modality.upper()}] {split}/{seq_dir.name}")
