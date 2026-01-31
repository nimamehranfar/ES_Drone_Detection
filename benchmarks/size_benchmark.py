#!/usr/bin/env python3
"""
Size / MIN_EVAL_AREA benchmark that:
- Runs Ultralytics 'yolo predict' ONCE at a low conf floor to export label txts (no images saved).
- Computes YOLO-val-like P/R/F1 at the BEST_F1 confidence ONLY for area=0 (baseline match).
- Uses that same fixed confidence for ALL other area thresholds (so FP cannot increase with threshold).
- Computes BOTH directions for each threshold:
    - GE: boxes with area >= threshold
    - LT: boxes with area < threshold
- Saves one append-only CSV (common across runs).
- Does NOT modify your dataset files (reads images/labels only; writes under TMP_RUN_DIR + CSV).

Notes:
- All areas are computed in ORIGINAL image pixel space (not 640x640 model space).
- Prediction txts from Ultralytics are normalized to original image size; we convert using per-image (w,h).
- Default matching IoU is 0.5 (Ultralytics default for detection metrics in val is IoU@0.50).
"""

import os
import csv
import json
import time
import shutil
import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import yaml
import cv2
import numpy as np
from tqdm import tqdm


# ==========================
# CONFIG (edit here)
# ==========================
YOLO_CMD = "yolo"

# If this script is in /benchmarks, PROJECT_ROOT is parent folder.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

MODEL_PATH = PROJECT_ROOT / "runs/detect/train7/weights/best.pt"
DATASET_YAML = PROJECT_ROOT / "datasets/drone_mixed.yaml"

DEVICE = "0"
IMGSZ = 640

# IMPORTANT:
# We run predict once with a low conf floor so we can later evaluate at any threshold
# and also reproduce val-style best-F1 selection for area=0.
PRED_CONF_FLOOR = 0.001

# NMS IoU used by predictor
PRED_IOU = 0.7  # Ultralytics default for predict is often 0.7; leave unless you intentionally change.

MAX_DET = 30
BATCH = 48

# Matching IoU threshold for evaluation (val-style IoU@0.50).
MATCH_IOU = 0.50

# Area thresholds to benchmark (pixel area in ORIGINAL image space)
MIN_EVAL_AREAS_PX = [0, 25, 100, 225, 400, 625, 900, 1600]

# How to treat predictions that overlap a GT that is EXCLUDED by the current slice:
# - "ignore_small_gt_for_fp": if an unmatched prediction overlaps ANY GT (excluded or not), ignore it (not FP).
#   This makes FP monotonic non-increasing as you increase the area threshold.
# - "subset": unmatched predictions are FP even if they overlap excluded GT.
FILTER_MODE = "ignore_small_gt_for_fp"  # keep this to prevent FP increases with threshold

# Output
BENCHMARK_DIR = PROJECT_ROOT / "benchmarks"
BENCHMARK_DIR.mkdir(exist_ok=True)

CSV_PATH = BENCHMARK_DIR / "min_eval_area_benchmark.csv"

# Temporary predict output (deleted at end unless KEEP_TMP_RUN_DIR=True)
TMP_RUN_DIR = BENCHMARK_DIR / "_tmp_predict"
KEEP_TMP_RUN_DIR = False

# Predictor output control
VERBOSE_CLI = False  # False => suppress Ultralytics per-image printing

# ==========================
# Helpers
# ==========================

def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def resolve_dataset_paths(dataset_yaml: Path, split: str = "test") -> Tuple[Path, Path, List[Path]]:
    """
    Returns:
      - images_dir
      - labels_dir
      - image_files list
    Works with Ultralytics YAML format:
      path: /root/dataset
      train: images/train
      val: images/val
      test: images/test
    """
    data = load_yaml(dataset_yaml)

    base = Path(data["path"]).expanduser()
    rel_images = Path(data[split])
    images_dir = (base / rel_images).resolve()

    # labels are usually: base/labels/<split_name>
    # If images_dir ends with .../images/test then labels should be .../labels/test
    split_name = images_dir.name
    labels_dir = images_dir.parent.parent / "labels" / split_name

    if not images_dir.exists():
        raise FileNotFoundError(f"Images dir not found: {images_dir}")
    if not labels_dir.exists():
        # Some datasets store labels in base/labels/test even if images path isn't .../images/test.
        # Try base/labels/<split> as fallback.
        fallback = base / "labels" / split
        if fallback.exists():
            labels_dir = fallback
        else:
            raise FileNotFoundError(f"Labels dir not found: {labels_dir} (and fallback {fallback} missing)")

    # Gather images with common extensions (Ultralytics supports many; keep it simple & correct)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    image_files = [p for p in images_dir.rglob("*") if p.suffix.lower() in exts]
    image_files.sort()

    if not image_files:
        raise FileNotFoundError(f"No images found under: {images_dir}")

    return images_dir, labels_dir, image_files

def yolo_norm_to_xyxy(norm_xywh: Tuple[float, float, float, float], img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    cx, cy, bw, bh = norm_xywh
    cx *= img_w
    cy *= img_h
    bw *= img_w
    bh *= img_h
    x1 = cx - bw / 2.0
    y1 = cy - bh / 2.0
    x2 = cx + bw / 2.0
    y2 = cy + bh / 2.0
    return (x1, y1, x2, y2)

def box_area_xyxy(b: Tuple[float, float, float, float]) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])

def iou_xyxy(b1: Tuple[float, float, float, float], b2: Tuple[float, float, float, float]) -> float:
    xA = max(b1[0], b2[0])
    yA = max(b1[1], b2[1])
    xB = min(b1[2], b2[2])
    yB = min(b1[3], b2[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    if inter <= 0.0:
        return 0.0
    a1 = box_area_xyxy(b1)
    a2 = box_area_xyxy(b2)
    denom = (a1 + a2 - inter)
    return inter / denom if denom > 0 else 0.0


@dataclass
class ImageRecord:
    img_path: Path
    label_path: Path
    w: int
    h: int
    gt_boxes_all: List[Tuple[float, float, float, float]]  # xyxy in original pixels
    pred_boxes_all: List[Tuple[float, float, float, float]]  # xyxy in original pixels
    pred_confs_all: List[float]


def read_image_hw(path: Path) -> Tuple[int, int]:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    h, w = img.shape[:2]
    return w, h

def read_gt_boxes(label_path: Path, img_w: int, img_h: int) -> List[Tuple[float, float, float, float]]:
    boxes = []
    if not label_path.exists():
        return boxes
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            # cls cx cy w h
            _, cx, cy, bw, bh = map(float, parts[:5])
            b = yolo_norm_to_xyxy((cx, cy, bw, bh), img_w, img_h)
            boxes.append(b)
    return boxes

def read_pred_boxes(pred_label_path: Path, img_w: int, img_h: int) -> Tuple[List[Tuple[float, float, float, float]], List[float]]:
    boxes, confs = [], []
    if not pred_label_path.exists():
        return boxes, confs
    with open(pred_label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            # cls cx cy w h conf
            _, cx, cy, bw, bh, conf = map(float, parts[:6])
            b = yolo_norm_to_xyxy((cx, cy, bw, bh), img_w, img_h)
            boxes.append(b)
            confs.append(conf)
    return boxes, confs


# ==========================
# Prediction (one-time)
# ==========================
def run_ultralytics_predict_once(images_dir: Path) -> Path:
    """
    Runs: yolo predict source=<images_dir> ...
    Returns path to labels folder produced by Ultralytics.
    """
    if TMP_RUN_DIR.exists():
        shutil.rmtree(TMP_RUN_DIR, ignore_errors=True)
    TMP_RUN_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        YOLO_CMD, "predict",
        f"model={MODEL_PATH}",
        f"source={images_dir}",
        f"imgsz={IMGSZ}",
        f"device={DEVICE}",
        f"conf={PRED_CONF_FLOOR}",
        f"iou={PRED_IOU}",
        f"max_det={MAX_DET}",
        f"batch={BATCH}",
        "save=False",
        "save_txt=True",
        "save_conf=True",
        f"project={TMP_RUN_DIR}",
        "name=run",
        f"verbose={str(VERBOSE_CLI)}",
    ]

    # Silence stdout if verbose is off to avoid massive per-image prints
    stdout = subprocess.DEVNULL if not VERBOSE_CLI else None
    stderr = subprocess.STDOUT if not VERBOSE_CLI else None

    subprocess.run(cmd, check=True, stdout=stdout, stderr=stderr)

    pred_labels_dir = TMP_RUN_DIR / "run" / "labels"
    if not pred_labels_dir.exists():
        raise FileNotFoundError(f"Ultralytics predict did not produce labels dir: {pred_labels_dir}")
    return pred_labels_dir


# ==========================
# Build index (images + GT + preds)
# ==========================
def build_index(split: str = "test") -> Tuple[List[ImageRecord], Path, Path]:
    images_dir, labels_dir, image_files = resolve_dataset_paths(DATASET_YAML, split=split)
    pred_labels_dir = run_ultralytics_predict_once(images_dir)

    records: List[ImageRecord] = []
    for img_path in tqdm(image_files, desc="Indexing images/GT/preds"):
        w, h = read_image_hw(img_path)
        label_path = labels_dir / f"{img_path.stem}.txt"

        gt_boxes = read_gt_boxes(label_path, w, h)

        pred_label_path = pred_labels_dir / f"{img_path.stem}.txt"
        pred_boxes, pred_confs = read_pred_boxes(pred_label_path, w, h)

        records.append(
            ImageRecord(
                img_path=img_path,
                label_path=label_path,
                w=w, h=h,
                gt_boxes_all=gt_boxes,
                pred_boxes_all=pred_boxes,
                pred_confs_all=pred_confs,
            )
        )
    return records, images_dir, labels_dir


# ==========================
# Evaluation at fixed confidence
# ==========================
@dataclass
class SliceStats:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    ignored_fp: int = 0  # preds ignored due to overlap with excluded GT (when FILTER_MODE=ignore...)
    images: int = 0
    img_tp: int = 0
    img_fp: int = 0
    img_tn: int = 0
    img_fn: int = 0

def eval_one_image(
    rec: ImageRecord,
    area_thr: float,
    conf_thr: float,
    slice_mode: str,  # "ge" or "lt"
) -> Tuple[int, int, int, int, int, int, int, int, int]:
    """
    Returns:
      tp, fp, fn, ignored_fp, img_tp, img_fp, img_tn, img_fn, images(=1)
    """
    assert slice_mode in ("ge", "lt")

    gt_all = rec.gt_boxes_all
    gt_all_areas = [box_area_xyxy(b) for b in gt_all]

    def in_slice(area: float) -> bool:
        return (area >= area_thr) if slice_mode == "ge" else (area < area_thr)

    target_gt = [b for b, a in zip(gt_all, gt_all_areas) if in_slice(a)]
    # excluded GT = all GT not in target
    excluded_gt = [b for b, a in zip(gt_all, gt_all_areas) if not in_slice(a)]

    # Filter preds by conf + slice
    preds = []
    for b, c in zip(rec.pred_boxes_all, rec.pred_confs_all):
        if c < conf_thr:
            continue
        a = box_area_xyxy(b)
        if in_slice(a):
            preds.append((b, c))

    # Sort preds by confidence desc (val-like greedy matching)
    preds.sort(key=lambda x: x[1], reverse=True)

    matched_gt = set()
    tp = fp = ignored_fp = 0

    for pb, pc in preds:
        best_iou = 0.0
        best_idx = -1
        for gi, gb in enumerate(target_gt):
            if gi in matched_gt:
                continue
            v = iou_xyxy(pb, gb)
            if v > best_iou:
                best_iou = v
                best_idx = gi

        if best_iou >= MATCH_IOU and best_idx >= 0:
            tp += 1
            matched_gt.add(best_idx)
        else:
            # Unmatched prediction: decide FP vs ignore based on FILTER_MODE
            if FILTER_MODE == "ignore_small_gt_for_fp" and area_thr > 0:
                # If it overlaps ANY GT (target or excluded) at IoU threshold -> ignore (not FP).
                any_match = False
                for gb in gt_all:
                    if iou_xyxy(pb, gb) >= MATCH_IOU:
                        any_match = True
                        break
                if any_match:
                    ignored_fp += 1
                else:
                    fp += 1
            else:
                fp += 1

    fn = len(target_gt) - len(matched_gt)

    # Image-level confusion (binary presence for this slice)
    gt_pos = len(target_gt) > 0
    # Pred positive if any real (tp/fp) pred exists; ignored preds don't count as positive
    pred_pos = (tp + fp) > 0

    img_tp = 1 if (gt_pos and pred_pos) else 0
    img_fp = 1 if ((not gt_pos) and pred_pos) else 0
    img_tn = 1 if ((not gt_pos) and (not pred_pos)) else 0
    img_fn = 1 if (gt_pos and (not pred_pos)) else 0

    return tp, fp, fn, ignored_fp, img_tp, img_fp, img_tn, img_fn, 1

def eval_slice(records: List[ImageRecord], area_thr: float, conf_thr: float, slice_mode: str) -> SliceStats:
    st = SliceStats()
    for rec in records:
        tp, fp, fn, ig, itp, ifp, itn, ifn, n = eval_one_image(rec, area_thr, conf_thr, slice_mode)
        st.tp += tp
        st.fp += fp
        st.fn += fn
        st.ignored_fp += ig
        st.img_tp += itp
        st.img_fp += ifp
        st.img_tn += itn
        st.img_fn += ifn
        st.images += n
    return st

def prf_from_stats(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
    return p, r, f1

def acc_from_image_stats(img_tp: int, img_fp: int, img_tn: int, img_fn: int) -> float:
    total = img_tp + img_fp + img_tn + img_fn
    return (img_tp + img_tn) / total if total else 0.0

def best_f1_conf_for_area0(records: List[ImageRecord]) -> Tuple[float, float, float, float]:
    """
    Computes best-F1 confidence threshold for area=0 (GE slice).
    We search over unique prediction confidences.
    Returns: best_conf, best_p, best_r, best_f1
    """
    # Collect all confs present across dataset
    confs = []
    for rec in records:
        confs.extend(rec.pred_confs_all)
    confs = sorted(set([float(c) for c in confs]), reverse=True)

    if not confs:
        return 0.25, 0.0, 0.0, 0.0

    # To match val behavior, include 0.0 boundary too
    # but confs already include low floor. We'll ensure minimum.
    if confs[-1] > 0.0:
        confs.append(0.0)

    best_conf = confs[0]
    best_p = best_r = best_f1 = -1.0

    # Evaluate with a progress bar (can be big; decimate if extremely large)
    # If there are too many unique confs, sample evenly to keep runtime reasonable.
    MAX_STEPS = 300
    if len(confs) > MAX_STEPS:
        idxs = np.linspace(0, len(confs) - 1, MAX_STEPS).astype(int).tolist()
        confs_eval = [confs[i] for i in idxs]
    else:
        confs_eval = confs

    for c in tqdm(confs_eval, desc="Searching best_conf (area=0)"):
        st = eval_slice(records, area_thr=0.0, conf_thr=c, slice_mode="ge")
        p, r, f1 = prf_from_stats(st.tp, st.fp, st.fn)
        if f1 > best_f1:
            best_f1 = f1
            best_p = p
            best_r = r
            best_conf = c

    return float(best_conf), float(best_p), float(best_r), float(best_f1)


# ==========================
# Main benchmark loop
# ==========================
def run_benchmark(split: str = "test") -> List[dict]:
    t0 = time.time()
    records, images_dir, labels_dir = build_index(split=split)
    index_sec = time.time() - t0

    best_conf0, p0, r0, f10 = best_f1_conf_for_area0(records)

    rows: List[dict] = []
    prev_fp_ge = None

    for area_thr in MIN_EVAL_AREAS_PX:
        start = time.time()

        # GE slice
        st_ge = eval_slice(records, area_thr=float(area_thr), conf_thr=best_conf0, slice_mode="ge")
        p_ge, r_ge, f1_ge = prf_from_stats(st_ge.tp, st_ge.fp, st_ge.fn)
        acc_ge = acc_from_image_stats(st_ge.img_tp, st_ge.img_fp, st_ge.img_tn, st_ge.img_fn)

        # LT slice (reverse)
        st_lt = eval_slice(records, area_thr=float(area_thr), conf_thr=best_conf0, slice_mode="lt")
        p_lt, r_lt, f1_lt = prf_from_stats(st_lt.tp, st_lt.fp, st_lt.fn)
        acc_lt = acc_from_image_stats(st_lt.img_tp, st_lt.img_fp, st_lt.img_tn, st_lt.img_fn)

        runtime = time.time() - start

        # Monotonic sanity (GE FP should not increase with threshold if conf is fixed and FILTER_MODE is ignore)
        fp_increase = 0
        if prev_fp_ge is not None and st_ge.fp > prev_fp_ge:
            fp_increase = 1
        prev_fp_ge = st_ge.fp

        row = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": str(MODEL_PATH),
            "dataset_yaml": str(DATASET_YAML),
            "split": split,
            "imgsz": IMGSZ,
            "device": DEVICE,
            "predict_conf_floor": PRED_CONF_FLOOR,
            "predict_iou": PRED_IOU,
            "predict_max_det": MAX_DET,
            "predict_batch": BATCH,
            "match_iou": MATCH_IOU,
            "filter_mode": FILTER_MODE,

            "best_conf_area0": round(best_conf0, 6),
            "area_thr_px": int(area_thr),

            # Baseline (area=0 best-F1) reference
            "area0_bestF1_precision": round(p0, 6),
            "area0_bestF1_recall": round(r0, 6),
            "area0_bestF1_f1": round(f10, 6),

            # >= threshold (main)
            "precision_ge": round(p_ge, 6),
            "recall_ge": round(r_ge, 6),
            "f1_ge": round(f1_ge, 6),
            "tp_ge": st_ge.tp,
            "fp_ge": st_ge.fp,
            "fn_ge": st_ge.fn,
            "ignored_fp_ge": st_ge.ignored_fp,

            "img_accuracy_ge": round(acc_ge, 6),
            "img_tp_ge": st_ge.img_tp,
            "img_fp_ge": st_ge.img_fp,
            "img_tn_ge": st_ge.img_tn,
            "img_fn_ge": st_ge.img_fn,
            "images_total": st_ge.images,

            # < threshold (reverse)
            "precision_lt": round(p_lt, 6),
            "recall_lt": round(r_lt, 6),
            "f1_lt": round(f1_lt, 6),
            "tp_lt": st_lt.tp,
            "fp_lt": st_lt.fp,
            "fn_lt": st_lt.fn,
            "ignored_fp_lt": st_lt.ignored_fp,

            "img_accuracy_lt": round(acc_lt, 6),
            "img_tp_lt": st_lt.img_tp,
            "img_fp_lt": st_lt.img_fp,
            "img_tn_lt": st_lt.img_tn,
            "img_fn_lt": st_lt.img_fn,

            # runtime info
            "index_build_sec": round(index_sec, 2),
            "eval_runtime_sec": round(runtime, 2),
            "fp_increase_flag_ge": fp_increase,
        }

        rows.append(row)

        print(
            f"area={area_thr:>5} | conf(fixed)={best_conf0:.4f} | "
            f"GE P/R/F1={p_ge:.3f}/{r_ge:.3f}/{f1_ge:.3f} (TP/FP/FN={st_ge.tp}/{st_ge.fp}/{st_ge.fn}) | "
            f"LT P/R/F1={p_lt:.3f}/{r_lt:.3f}/{f1_lt:.3f} (TP/FP/FN={st_lt.tp}/{st_lt.fp}/{st_lt.fn})"
        )

    return rows


def append_csv(rows: List[dict], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = parser.parse_args()

    rows = run_benchmark(split=args.split)
    append_csv(rows, CSV_PATH)

    if not KEEP_TMP_RUN_DIR and TMP_RUN_DIR.exists():
        shutil.rmtree(TMP_RUN_DIR, ignore_errors=True)

    print(f"\nSaved results to: {CSV_PATH}")


if __name__ == "__main__":
    main()
