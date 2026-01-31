"""
Drone Detection Demo

Frame-based YOLO drone detector with optional Temporal ROI Propagation, Cascaded ROI Confirmation,
temporal continuity (decision-only), and warning/alert windows. Designed for repeatable drone detection.
"""

import os
import time
import csv
import cv2
import numpy as np
from collections import deque, defaultdict
from datetime import datetime
import uuid
from typing import List, Tuple, Optional, Dict
from pathlib import Path
from ultralytics import YOLO

# --- Paths ---
MODEL_WEIGHTS = "Yolo26n_trained/weights/best.pt"
VIDEO_PATH = "video_test/dji_mavick_distant_hillside.avi"

# Annotation Logic
ANNOTATION_PATH_PRIMARY = ""

# Fallback: same path as video, just .txt instead of .mp4
ANNOTATION_PATH_FALLBACK = str(Path(VIDEO_PATH).with_suffix(".txt"))
ANNOTATION_PATH = ANNOTATION_PATH_PRIMARY if os.path.exists(ANNOTATION_PATH_PRIMARY) else ANNOTATION_PATH_FALLBACK

OUTPUT_ROOT = "demo_outputs"

# --- Inference Policy ---
INFER_FPS = 5  # Target YOLO inference calls per second (stride-based)
ROI_SIZE = 640  # Square crop size for Temporal ROI Propagation + Cascaded ROI Confirmation ROI
BASE_IMGSZ = 640

# --- Detection Constraints ---
MAX_FULLFRAME_DETECTIONS = 3
MAX_GUIDED_ROIS = 3
MAX_ROI_DETECTIONS = 1  # For any ROI inference : max 1 detection

# --- Confidence Thresholds (independent) ---
DETECT_CONF = 0.25  # main YOLO conf (full-frame)
TROI_DETECT_CONF = 0.25  # main YOLO conf (temporal ROI propagation). Keep same as DETECT_CONF for fair comparisons.
CASCADE_TRIGGER_CONF = 0.40  # triggers cascaded ROI confirmation (Confirm Low/Small mode)

# Cascaded ROI Confirmation: two-step thresholding (not redundant!)
# - CASCADE_DETECT_CONF controls which boxes YOLO even returns (speed/volume).
# - CASCADE_ACCEPT_CONF is the acceptance rule for keeping the detection (quality).
CASCADE_DETECT_CONF = 0.25  # YOLO conf inside cascaded ROI confirmation ROI pass (keep low to avoid "no-box" failures)
CASCADE_ACCEPT_CONF = 0.40  # required cascaded ROI confirmation confidence to ACCEPT (otherwise detection is dropped)

# --- Size / Area Thresholds (ALL are AREAS in pixels) ---
# Global evaluation filter: if >0, ignore both GT and predictions smaller than this area.
# Example: 10*10 = 100 means ignore anything <100px².
MIN_EVAL_AREA_PX2 = 0

# Temporal ROI Propagation eligibility (area threshold, in full-frame pixels)
MIN_TROI_SEED_AREA_PX2 = 15 * 15

# Cascaded ROI Confirmation trigger for "Cascade Low/Small" mode:
# if area < CASCADE_TRIGGER_AREA_PX2 OR main_conf < CASCADE_TRIGGER_CONF -> verify.
CASCADE_TRIGGER_AREA_PX2 = 25 * 25

# Alert decision thresholds (area-based)
ALERT_MIN_AREA_PX2 = 25 * 25  # base alert size (no cascaded confirmation)
ALERT_MIN_AREA_CASCADED_PX2 = 15 * 15  # if cascaded confirmation enabled, verified smaller drones can alert

# --- Warning / Alert Logic (INFERENCE-FRAME BASED) ---
WARNING_WINDOW_FRAMES = 10
WARNING_REQUIRE_HITS = 9

ALERT_WINDOW_FRAMES = 10
ALERT_REQUIRE_HITS = 9

# Cooldowns (seconds). After an event fires, its window resets and stays idle for this long.
WARNING_COOLDOWN_S = 3.0
ALERT_COOLDOWN_S = 3.0

# --- Cascaded ROI Confirmation modes (no widgets) ---
# Options:
#   "None"
#   "Cascade Low/Small"
#   "Cascade All"
#   "Alert-Window Cascade"  (cascaded ROI confirmation only when alert is about to trigger; affects alert vs warning only)
CASCADED_ROI_CONFIRM_MODE = "None"

# Alert-window verify params (only used in "Alert-Window Cascade")
ALERTWIN_CASCADE_DETECT_CONF = 0.25
ALERTWIN_CASCADE_ACCEPT_CONF = 0.40
ALERTWIN_CASCADE_AVGCONF_ACCEPT = 0.5  # average of accepted verify confs in the window
ALERTWIN_CASCADE_MIN_ACCEPTS = 0  # how many of the window frames must verify to accept alert

# --- Temporal ROI Propagation toggle ---
TEMPORAL_ROI_PROP_ENABLED = False  # Temporal ROI Propagation

# --- Overlay toggles ---
SHOW_GT = False
SHOW_GATE = False
SHOW_TROI = False
SHOW_CASCADE = False

TOPLEFT_LOG_MODE = "windows_big"  # "off" | "full" | "windows_big"
SHOW_SOURCE_TAGS = False  # show small source tags on prediction boxes (FULL/GROI/VER)

# --- Real-time viewing ---
SHOW_WINDOW = True
PAUSE_KEY = 'p'  # press to pause/resume when SHOW_WINDOW=True
WINDOW_NAME = "Drone Detection Demo"
WINDOW_FIT_TO_SCREEN = True
WINDOW_SCALE = 0.9  # fraction of screen (fit-to-window)

# --- Output ---
SAVE_VIDEO = False

# Save alert-window frames (10 inference frames) when window reaches ALERT_REQUIRE_HITS.
# Saves images with ONLY the detected drone box (no logs/ROIs/gates). Works for any mode.
SAVE_ALERT_WINDOW_FRAMES = True
ALERT_WINDOW_SAVE_SUBDIR = "saved_alert_windows"

DRONE_LABEL = "drone"


def xywh_to_xyxy(x, y, w, h):
    return (x, y, x + w, y + h)


def box_area(b):
    return max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))


def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1);
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2);
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    union = box_area(a) + box_area(b) - inter
    return float(inter / union) if union > 0 else 0.0


def center_of(b):
    return ((float(b[0]) + float(b[2])) / 2.0, (float(b[1]) + float(b[3])) / 2.0)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def expand_box_from_center(b, factor, W, H):
    cx, cy = center_of(b)
    w = float(b[2]) - float(b[0])
    h = float(b[3]) - float(b[1])
    nw = w * float(factor)
    nh = h * float(factor)
    x1 = int(cx - nw / 2)
    y1 = int(cy - nh / 2)
    x2 = int(cx + nw / 2)
    y2 = int(cy + nh / 2)
    return (clamp(x1, 0, W), clamp(y1, 0, H), clamp(x2, 0, W), clamp(y2, 0, H))


def crop_square_around_point(frame, cx, cy, size):
    H, W = frame.shape[:2]
    half = size // 2
    x1 = clamp(int(cx - half), 0, W - 1)
    y1 = clamp(int(cy - half), 0, H - 1)
    x2 = clamp(int(x1 + size), 1, W)
    y2 = clamp(int(y1 + size), 1, H)
    # ensure at most size and keep square-ish
    if x2 - x1 < size:
        if x1 == 0:
            x2 = min(W, x1 + size)
        else:
            x1 = max(0, x2 - size)
    if y2 - y1 < size:
        if y1 == 0:
            y2 = min(H, y1 + size)
        else:
            y1 = max(0, y2 - size)
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


def draw_box(img, b, color, thickness=2, label=None, anchor="tl"):
    """Draw rectangle and a label OUTSIDE the box to avoid overlap with edges.

    anchor:
      - 'tl' top-left (label above box), 'tr' top-right (label above box)
      - 'bl' bottom-left (label below box), 'br' bottom-right (label below box)
    """
    x1, y1, x2, y2 = map(int, b)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if not label:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thick = 1
    (tw, th), base = cv2.getTextSize(label, font, scale, thick)

    H, W = img.shape[:2]
    pad = 2

    # Default placements (outside)
    if anchor == "tr":
        tx = x2 - tw - pad
        ty = y1 - pad
    elif anchor == "bl":
        tx = x1 + pad
        ty = y2 + th + pad
    elif anchor == "br":
        tx = x2 - tw - pad
        ty = y2 + th + pad
    else:  # "tl"
        tx = x1 + pad
        ty = y1 - pad

    # Keep on-screen while staying outside if possible
    tx = int(clamp(tx, 0, max(0, W - tw - 1)))

    # If above the image, flip to below the box
    if ty - th < 0:
        ty = y2 + th + pad
    # If below the image, flip to above the box
    if ty >= H:
        ty = y1 - pad
    # Final clamp
    ty = int(clamp(ty, th + 1, H - 2))

    cv2.putText(img, label, (tx, ty), font, scale, color, thick, cv2.LINE_AA)


def overlay_text(img, lines, x=10, y=20, line_h=20):
    for i, t in enumerate(lines):
        yy = y + i * line_h
        cv2.putText(img, t, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, t, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def overlay_text_big(img, lines, x=20, y=60, line_h=48, scale=1.35):
    """Big-font overlay for TOPLEFT_LOG_MODE='windows_big'."""
    for i, t in enumerate(lines):
        yy = y + i * line_h
        cv2.putText(img, t, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, float(scale), (0, 0, 0), 6, cv2.LINE_AA)
        cv2.putText(img, t, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, float(scale), (255, 255, 255), 2, cv2.LINE_AA)


def load_annotations(path, label_name):
    """Expected line format:
    frame_id 0
    frame_id 1 x y w h label
    Keeps up to 3 GT boxes per frame.
    """
    gt = defaultdict(list)
    if not os.path.exists(path):
        return gt
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            p = line.strip().split()
            if len(p) < 2:
                continue
            fid = int(p[0])
            if p[1] == "0":
                if fid not in gt:
                    gt[fid] = []
                continue
            if len(p) >= 7 and p[6] == label_name:
                x, y, w, h = map(float, p[2:6])
                gt[fid].append(xywh_to_xyxy(x, y, w, h))
    for k in list(gt.keys()):
        if len(gt[k]) > 3:
            gt[k] = gt[k][:3]
    return gt


def nms_indices_iou(boxes, confs, iou_thresh=0.45):
    """Return kept indices in descending confidence order using IoU suppression."""
    if not boxes:
        return []
    idxs = np.argsort(np.array(confs))[::-1]
    keep = []
    while len(idxs) > 0:
        i = int(idxs[0])
        keep.append(i)
        if len(idxs) == 1:
            break
        rest = idxs[1:]
        suppressed = []
        for j in rest:
            if iou_xyxy(boxes[i], boxes[int(j)]) > iou_thresh:
                suppressed.append(int(j))
        idxs = np.array([j for j in rest if int(j) not in suppressed], dtype=int)
    return keep


def compute_map50_greedy_iou50(inference_rows):
    """Compute mAP@0.50 on inference frames only (greedy matching, up to 3 GT/preds per frame)."""
    all_predictions = []  # (confidence, frame_id, box_xyxy)
    total_gt = 0

    for row in inference_rows:
        fid = int(row["frame_id"])
        gt_boxes = row.get("gt_boxes", [])
        pred_boxes = row.get("pred_boxes", [])
        pred_confs = row.get("pred_confs", [])

        total_gt += len(gt_boxes)
        for pb, pc in zip(pred_boxes, pred_confs):
            all_predictions.append((float(pc), fid, pb))

    if total_gt == 0:
        return 0.0

    all_predictions.sort(key=lambda x: x[0], reverse=True)

    matched = set()  # (frame_id, gt_idx)
    tps = []
    fps = []

    gt_by_frame = {int(r["frame_id"]): r.get("gt_boxes", []) for r in inference_rows}

    for conf, fid, pb in all_predictions:
        gts = gt_by_frame.get(fid, [])
        best_iou = 0.0
        best_gt_idx = -1
        for i, gb in enumerate(gts):
            if (fid, i) in matched:
                continue
            iou = iou_xyxy(pb, gb)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i
        if best_iou >= 0.5 and best_gt_idx >= 0:
            matched.add((fid, best_gt_idx))
            tps.append(1)
            fps.append(0)
        else:
            tps.append(0)
            fps.append(1)

    tps_cum = np.cumsum(tps)
    fps_cum = np.cumsum(fps)
    precisions = tps_cum / np.maximum(tps_cum + fps_cum, 1e-9)
    recalls = tps_cum / float(total_gt)

    ap = 0.0
    for r in np.linspace(0, 1, 11):
        p = np.max(precisions[recalls >= r]) if np.any(recalls >= r) else 0.0
        ap += p / 11.0
    return float(ap)


def get_screen_size():
    """Best-effort screen resolution without extra dependencies."""
    # tkinter is stdlib; use if available (works on Windows/Linux/macOS in most setups)
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        w = root.winfo_screenwidth()
        h = root.winfo_screenheight()
        root.destroy()
        return int(w), int(h)
    except Exception:
        return None


def fit_size_to_screen(img_w, img_h, screen_w, screen_h, scale):
    max_w = int(screen_w * scale)
    max_h = int(screen_h * scale)
    if img_w <= 0 or img_h <= 0:
        return img_w, img_h
    s = min(max_w / img_w, max_h / img_h, 1.0)
    return max(1, int(img_w * s)), max(1, int(img_h * s))


class TemporalContinuity:
    """
    Temporal continuity is ONLY for counting hit-frames inside Warning/Alert confirmation windows.
    Does NOT affect detection metrics.
    """

    def __init__(self, stride: int):
        self.stride = max(1, int(stride))
        self.prev_gate: Optional[Tuple[float, float, float, float]] = None
        self.prev_had_det: bool = False

    def reset(self):
        self.prev_gate = None
        self.prev_had_det = False

    def expansion_factor(self):
        # stride=1 -> ~2x, larger stride -> larger expansion (clamped)
        return float(clamp(2.0 + 0.15 * (self.stride - 1), 2.0, 6.0))

    def build_gate(self, det_box, W, H):
        return expand_box_from_center(det_box, self.expansion_factor(), W, H)

    def accept(self, det_box, W, H):
        curr_gate = self.build_gate(det_box, W, H)
        if not self.prev_had_det or self.prev_gate is None:
            self.prev_gate = curr_gate
            self.prev_had_det = True
            return True, curr_gate, None, True  # first det after reset accepted
        a = self.prev_gate
        b = curr_gate
        ix1 = max(a[0], b[0]);
        iy1 = max(a[1], b[1])
        ix2 = min(a[2], b[2]);
        iy2 = min(a[3], b[3])
        inter = (ix2 - ix1) > 0 and (iy2 - iy1) > 0
        accepted = bool(inter)
        self.prev_gate = curr_gate
        self.prev_had_det = True
        return accepted, curr_gate, a, False


def build_model():
    model = YOLO(MODEL_WEIGHTS)
    names = model.model.names
    drone_cls_id = 0
    for k, v in names.items():
        if str(v).lower() == str(DRONE_LABEL).lower():
            drone_cls_id = int(k)
            break
    return model, names, drone_cls_id


def infer_roi_and_project_to_frame(model, drone_cls_id, img_bgr, roi_rect, conf, max_det):
    """Runs YOLO on img_bgr and maps detections back into full-frame coords using roi_rect offset."""
    rx, ry, _, _ = roi_rect
    res = model.predict(img_bgr, conf=float(conf), max_det=int(max_det), verbose=False, imgsz=int(BASE_IMGSZ))[0]
    boxes, confs = [], []
    if res.boxes is not None and len(res.boxes) > 0:
        xyxy = res.boxes.xyxy.cpu().numpy()
        cnf = res.boxes.conf.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy()
        for i in range(len(xyxy)):
            if int(cls[i]) == int(drone_cls_id):
                x1, y1, x2, y2 = xyxy[i]
                boxes.append((float(x1 + rx), float(y1 + ry), float(x2 + rx), float(y2 + ry)))
                confs.append(float(cnf[i]))
    return boxes, confs


def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    GT_DATA = load_annotations(ANNOTATION_PATH, DRONE_LABEL)
    print(f"Loaded GT for {len(GT_DATA)} frames from: {ANNOTATION_PATH}")

    model, names, DRONE_CLASS_ID = build_model()
    print("DRONE_CLASS_ID =", DRONE_CLASS_ID, "names =", names)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open VIDEO_PATH: {VIDEO_PATH}")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) else -1

    stride = max(1, int(round(fps / float(INFER_FPS))))
    print(f"Res: {W}x{H}, FPS: {fps:.2f}, Stride: {stride}, Frames: {total_frames}")

    ROI_SIZE_LOCAL = int(ROI_SIZE)

    # Unique run id (for unique output videos)
    RUN_ID = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    val_pre = bool(TEMPORAL_ROI_PROP_ENABLED)
    val_post_mode = str(CASCADED_ROI_CONFIRM_MODE)

    val_show_gt = bool(SHOW_GT)
    val_show_gate = bool(SHOW_GATE)
    val_show_guided = bool(SHOW_TROI)
    val_show_verify = bool(SHOW_CASCADE)

    # Video output
    vid_out_path = None
    writer = None
    if SAVE_VIDEO:
        base = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
        out_name = f"bench_{base}_{RUN_ID}_infer{int(INFER_FPS)}.mp4"
        vid_out_path = os.path.join(OUTPUT_ROOT, out_name)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(vid_out_path, fourcc, float(fps), (W, H))

    # Real-time window (resizable, fit to screen)
    screen = get_screen_size() if (SHOW_WINDOW and WINDOW_FIT_TO_SCREEN) else None
    disp_w, disp_h = W, H
    if screen is not None:
        disp_w, disp_h = fit_size_to_screen(W, H, screen[0], screen[1], float(WINDOW_SCALE))
    if SHOW_WINDOW:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        try:
            cv2.resizeWindow(WINDOW_NAME, disp_w, disp_h)
        except Exception:
            pass

    # Temporal continuitys for prediction-based warning/alert
    temporal_warning = TemporalContinuity(stride)
    temporal_alert = TemporalContinuity(stride)
    win_warning = deque(maxlen=int(WARNING_WINDOW_FRAMES))
    win_alert = deque(maxlen=int(ALERT_WINDOW_FRAMES))

    # Alert-window frame saving (log-only artifact)
    alert_save_buf = deque(maxlen=int(ALERT_WINDOW_FRAMES))  # items: (frame_bgr, rep_box, frame_id)
    last_alert_pre_state = False

    # Alert-window cascade status line persistence (in VIDEO frames)
    alertwin_cascade_status_msg = None
    alertwin_cascade_status_ttl = 0

    warning_active = False
    alert_active = False
    count_alert_events = 0
    count_warning_events = 0
    last_alert_state = False
    last_warning_state = False

    # Cooldowns (in inference frames)
    warn_cooldown_frames = int(round(float(WARNING_COOLDOWN_S) * float(INFER_FPS)))
    alert_cooldown_frames = int(round(float(ALERT_COOLDOWN_S) * float(INFER_FPS)))
    warn_cooldown_left = 0
    alert_cooldown_left = 0

    frame_id = 0
    inference_rows = []

    # Counters and timing
    total_infer_time_s = 0.0
    alertwin_cascade_time_s = 0.0  # extra Cascaded ROI Confirmation time in 'Alert-Window Cascade' (already included in total_infer_time_s)
    num_infer_frames = 0
    total_yolo_calls = 0

    cnt_full = 0
    cnt_guided = 0
    cnt_verify_pass = 0
    cnt_verify_fail = 0

    # Cascaded ROI Confirmation-triggered confidence tracking
    conf_verify_trigger_main_all = []
    conf_verify_trigger_post_any = []
    conf_verify_pass_main = []
    conf_verify_pass_post = []
    conf_verify_fail_main = []
    conf_verify_fail_post_any = []

    # Overlay caching (no flicker on non-infer frames)
    last_infer_final_boxes = []
    last_infer_final_confs = []
    last_infer_final_sources = []
    last_infer_final_main_confs = []
    last_infer_final_post_confs = []

    last_infer_guided_rois = []
    last_infer_verify_rois = []

    last_infer_gate_prev_warn = None
    last_infer_gate_curr_warn = None
    last_infer_gate_prev_alert = None
    last_infer_gate_curr_alert = None
    last_infer_gate_warn_accepted = None
    last_infer_gate_alert_accepted = None

    last_infer_warn_state = False
    last_infer_alert_state = False
    last_infer_gt_warn_state = False
    last_infer_gt_alert_state = False

    last_infer_verify_stats = (0, 0, 0)  # tried, pass, fail
    last_infer_verify_details = []  # list of strings (includes rejected attempts)

    # Alert-window verify bookkeeping (only for "Alert-Window Cascade")
    alert_window_rep_boxes = deque(maxlen=int(ALERT_WINDOW_FRAMES))
    alert_window_rep_main_confs = deque(maxlen=int(ALERT_WINDOW_FRAMES))
    alert_window_rep_frame_ids = deque(maxlen=int(ALERT_WINDOW_FRAMES))
    alert_window_rep_frames = deque(maxlen=int(ALERT_WINDOW_FRAMES))

    alert_window_verify_triggers = 0
    alert_window_verify_accepts = 0
    alert_window_verify_rejects = 0
    alert_window_verify_avg_conf_last = None

    def pick_best_match_box(boxes, last_box):
        if not boxes:
            return None
        if last_box is None:
            return boxes[0]
        best_i = -1
        best = 0.0
        for i, b in enumerate(boxes):
            v = iou_xyxy(b, last_box)
            if v > best:
                best = v
                best_i = i
        return boxes[best_i] if best_i >= 0 else boxes[0]

    def filter_dets_by_min_area_px2(boxes, confs, sources, main_confs, post_confs):
        if int(MIN_EVAL_AREA_PX2) <= 0:
            return boxes, confs, sources, main_confs, post_confs
        out_b, out_c, out_s, out_mc, out_pc = [], [], [], [], []
        for b, c, s, mc, pc in zip(boxes, confs, sources, main_confs, post_confs):
            if box_area(b) >= float(MIN_EVAL_AREA_PX2):
                out_b.append(b);
                out_c.append(c);
                out_s.append(s);
                out_mc.append(mc);
                out_pc.append(pc)
        return out_b, out_c, out_s, out_mc, out_pc

    # GT-based reference warning/alert
    gt_temporal_warning = TemporalContinuity(stride)
    gt_temporal_alert = TemporalContinuity(stride)
    gt_win_warning = deque(maxlen=int(WARNING_WINDOW_FRAMES))
    gt_win_alert = deque(maxlen=int(ALERT_WINDOW_FRAMES))
    gt_last_warning_state = False
    gt_last_alert_state = False
    gt_count_warning_events = 0
    gt_count_alert_events = 0
    gt_warn_cooldown_left = 0
    gt_alert_cooldown_left = 0

    # Frame-level agreement vs GT (inference frames only)
    warn_state_tp = warn_state_fp = warn_state_fn = warn_state_tn = 0
    alert_state_tp = alert_state_fp = alert_state_fn = alert_state_tn = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        is_infer = (frame_id % stride == 0)
        vis = frame.copy()

        # Persist alert-window cascade status line for a short time (no flicker)
        if alertwin_cascade_status_ttl > 0:
            alertwin_cascade_status_ttl -= 1
            if alertwin_cascade_status_ttl <= 0:
                alertwin_cascade_status_msg = None

        if is_infer:
            infer_t0 = time.perf_counter()
            num_infer_frames += 1

            # Ground truth (0..3), filter by MIN_EVAL_AREA_PX2 if enabled
            gt_boxes = GT_DATA.get(frame_id, [])[:3]
            if int(MIN_EVAL_AREA_PX2) > 0:
                gt_boxes = [b for b in gt_boxes if box_area(b) >= float(MIN_EVAL_AREA_PX2)]

            # ============================
            # Main inference (full frame OR temporal ROI propagation)
            # ============================
            final_boxes = []
            final_confs = []
            final_sources = []
            final_main_confs = []
            final_post_confs = []

            guided_rois = []
            verify_rois = []
            verify_details = []

            prev_boxes = last_infer_final_boxes[:] if last_infer_final_boxes else []
            eligible_prev = [b for b in prev_boxes if box_area(b) >= float(MIN_TROI_SEED_AREA_PX2)]

            used_guided = False
            if val_pre and len(eligible_prev) > 0:
                used_guided = True
                eligible_prev_sorted = sorted(eligible_prev, key=lambda b: box_area(b), reverse=True)[:int(MAX_GUIDED_ROIS)]
                for pb in eligible_prev_sorted:
                    cx, cy = center_of(pb)
                    crop, roi = crop_square_around_point(frame, cx, cy, ROI_SIZE_LOCAL)
                    guided_rois.append(roi)
                    boxes_roi, confs_roi = infer_roi_and_project_to_frame(
                        model, DRONE_CLASS_ID, crop, (roi[0], roi[1], roi[2], roi[3]),
                        conf=TROI_DETECT_CONF, max_det=MAX_ROI_DETECTIONS
                    )
                    total_yolo_calls += 1
                    if boxes_roi:
                        b = boxes_roi[0]
                        c = float(confs_roi[0]) if confs_roi else 0.0
                        final_boxes.append(b)
                        final_confs.append(c)
                        final_sources.append("troi")
                        final_main_confs.append(c)
                        final_post_confs.append(None)
                        cnt_guided += 1
                    if len(final_boxes) >= int(MAX_FULLFRAME_DETECTIONS):
                        break

            # If Temporal ROI Propagation is used this inference frame, skip full-frame inference.
            if not used_guided:
                boxes_full, confs_full = infer_roi_and_project_to_frame(
                    model, DRONE_CLASS_ID, frame, (0, 0, W, H),
                    conf=DETECT_CONF, max_det=MAX_FULLFRAME_DETECTIONS
                )
                total_yolo_calls += 1
                for b, c in zip(boxes_full, confs_full):
                    final_boxes.append(b)
                    final_confs.append(float(c))
                    final_sources.append("full")
                    final_main_confs.append(float(c))
                    final_post_confs.append(None)
                    cnt_full += 1

            # Filter by MIN_EVAL_AREA_PX2 (affects everything: metrics + continuity + events)
            final_boxes, final_confs, final_sources, final_main_confs, final_post_confs = filter_dets_by_min_area_px2(
                final_boxes, final_confs, final_sources, final_main_confs, final_post_confs
            )

            # Cap to max 3 predictions before cascaded confirmation
            if len(final_boxes) > int(MAX_FULLFRAME_DETECTIONS):
                final_boxes = final_boxes[:int(MAX_FULLFRAME_DETECTIONS)]
                final_confs = final_confs[:int(MAX_FULLFRAME_DETECTIONS)]
                final_sources = final_sources[:int(MAX_FULLFRAME_DETECTIONS)]
                final_main_confs = final_main_confs[:int(MAX_FULLFRAME_DETECTIONS)]
                final_post_confs = final_post_confs[:int(MAX_FULLFRAME_DETECTIONS)]

            # ============================
            # Cascaded ROI Confirmation (per-detection)
            # ============================
            verify_tried = 0
            verify_pass = 0
            verify_fail = 0

            def should_verify_area(b, main_conf):
                if val_post_mode == "None":
                    return False
                if val_post_mode == "Cascade All":
                    return True
                if val_post_mode == "Alert-Window Cascade":
                    return False  # cascaded ROI confirmation happens only for alert decision, not per-detection
                # Confirm Low/Small
                small = (box_area(b) < float(CASCADE_TRIGGER_AREA_PX2))
                low = (float(main_conf) < float(CASCADE_TRIGGER_CONF))
                return bool(small or low)

            verified_boxes = []
            verified_confs = []
            verified_sources = []
            verified_main_confs = []
            verified_post_confs = []

            for idx, (b, c, s, main_c, postc) in enumerate(zip(final_boxes, final_confs, final_sources, final_main_confs, final_post_confs), start=1):
                if should_verify_area(b, c):
                    verify_tried += 1
                    conf_verify_trigger_main_all.append(float(c))
                    cx, cy = center_of(b)
                    crop, roi = crop_square_around_point(frame, cx, cy, ROI_SIZE_LOCAL)
                    verify_rois.append(roi)

                    boxes_v, confs_v = infer_roi_and_project_to_frame(
                        model, DRONE_CLASS_ID, crop, (roi[0], roi[1], roi[2], roi[3]),
                        conf=CASCADE_DETECT_CONF, max_det=MAX_ROI_DETECTIONS
                    )
                    total_yolo_calls += 1

                    if boxes_v:
                        post_conf = float(confs_v[0]) if confs_v else 0.0
                        conf_verify_trigger_post_any.append(post_conf)
                        if post_conf >= float(CASCADE_ACCEPT_CONF):
                            verify_pass += 1
                            cnt_verify_pass += 1
                            conf_verify_pass_main.append(float(c))
                            conf_verify_pass_post.append(post_conf)

                            vb = boxes_v[0]
                            vc = post_conf
                            verified_boxes.append(vb)
                            verified_confs.append(vc)
                            verified_sources.append("cascade")
                            verified_main_confs.append(float(c))
                            verified_post_confs.append(post_conf)
                            verify_details.append(f"P{idx}: {c:.2f}->{post_conf:.2f} ACCEPT")
                        else:
                            verify_fail += 1
                            cnt_verify_fail += 1
                            conf_verify_fail_main.append(float(c))
                            conf_verify_fail_post_any.append(post_conf)
                            verify_details.append(f"P{idx}: {c:.2f}->{post_conf:.2f} REJECT")
                            # drop
                    else:
                        verify_fail += 1
                        cnt_verify_fail += 1
                        conf_verify_fail_main.append(float(c))
                        verify_details.append(f"P{idx}: {c:.2f}->NONE REJECT")
                        # drop
                else:
                    verified_boxes.append(b)
                    verified_confs.append(float(c))
                    verified_sources.append(s)
                    verified_main_confs.append(float(main_c))
                    verified_post_confs.append(None)

            # Filter by MIN_EVAL_AREA_PX2 again (cascaded ROI confirmation could shrink boxes)
            verified_boxes, verified_confs, verified_sources, verified_main_confs, verified_post_confs = filter_dets_by_min_area_px2(
                verified_boxes, verified_confs, verified_sources, verified_main_confs, verified_post_confs
            )

            # NMS across final set
            keep_idx = nms_indices_iou(verified_boxes, verified_confs, iou_thresh=0.45)
            verified_boxes = [verified_boxes[i] for i in keep_idx]
            verified_confs = [verified_confs[i] for i in keep_idx]
            verified_sources = [verified_sources[i] for i in keep_idx]
            verified_main_confs = [verified_main_confs[i] for i in keep_idx]
            verified_post_confs = [verified_post_confs[i] for i in keep_idx]

            if len(verified_boxes) > int(MAX_FULLFRAME_DETECTIONS):
                verified_boxes = verified_boxes[:int(MAX_FULLFRAME_DETECTIONS)]
                verified_confs = verified_confs[:int(MAX_FULLFRAME_DETECTIONS)]
                verified_sources = verified_sources[:int(MAX_FULLFRAME_DETECTIONS)]
                verified_main_confs = verified_main_confs[:int(MAX_FULLFRAME_DETECTIONS)]
                verified_post_confs = verified_post_confs[:int(MAX_FULLFRAME_DETECTIONS)]

            # ============================
            # Temporal continuity + Warning/Alert counting (prediction-based)
            # ============================
            rep_box = pick_best_match_box(
                verified_boxes,
                last_infer_final_boxes[0] if last_infer_final_boxes else None
            )
            rep_main_conf = None
            if rep_box is not None:
                # best effort: use first box's main conf (aligned by order after NMS)
                rep_main_conf = float(verified_main_confs[0]) if verified_main_confs else None

            # Buffer inference frames for optional alert-window saving
            if bool(SAVE_ALERT_WINDOW_FRAMES) and alert_cooldown_left <= 0:
                alert_save_buf.append((frame.copy(), rep_box, int(frame_id)))

            warn_hit = False
            alert_hit = False
            warn_gate_prev = None
            warn_gate_curr = None
            warn_acc = None
            alert_gate_prev = None
            alert_gate_curr = None
            alert_acc = None

            # Cooldown handling: during cooldown we do not accumulate windows at all
            if warn_cooldown_left > 0:
                warn_cooldown_left -= 1
                temporal_warning.reset()
                win_warning.clear()
                warning_active = False
                warn_acc = None
            if alert_cooldown_left > 0:
                alert_cooldown_left -= 1
                temporal_alert.reset()
                win_alert.clear()
                alert_active = False
                alert_acc = None
                # keep alert-window buffers cleared too
                alert_window_rep_boxes.clear()
                alert_window_rep_main_confs.clear()
                alert_window_rep_frame_ids.clear()
                alert_window_rep_frames.clear()

            if rep_box is None:
                temporal_warning.reset()
                temporal_alert.reset()
                win_warning.clear()
                win_alert.clear()
                alert_window_rep_boxes.clear()
                alert_window_rep_main_confs.clear()
                alert_window_rep_frame_ids.clear()
                alert_window_rep_frames.clear()
                warn_hit = False
                alert_hit = False
            else:
                # Warning: any size, continuity-gated
                ok_w, curr_w, prev_w, _ = temporal_warning.accept(rep_box, W, H)
                warn_acc = bool(ok_w)
                warn_gate_curr = curr_w
                warn_gate_prev = prev_w
                warn_hit = bool(ok_w) if warn_cooldown_left <= 0 else False

                # Alert: size rule (area-based)
                det_area = box_area(rep_box)
                any_verified_present = ("cascade" in verified_sources)

                if val_post_mode in ("None", "Alert-Window Cascade"):
                    size_ok = (det_area >= float(ALERT_MIN_AREA_PX2))
                else:
                    size_ok = (det_area >= float(ALERT_MIN_AREA_PX2)) or (
                            det_area >= float(ALERT_MIN_AREA_CASCADED_PX2) and any_verified_present
                    )

                if size_ok and alert_cooldown_left <= 0:
                    ok_a, curr_a, prev_a, _ = temporal_alert.accept(rep_box, W, H)
                    alert_acc = bool(ok_a)
                    alert_gate_curr = curr_a
                    alert_gate_prev = prev_a
                    alert_hit = bool(ok_a)
                else:
                    temporal_alert.reset()
                    win_alert.clear()
                    alert_window_rep_boxes.clear()
                    alert_window_rep_main_confs.clear()
                    alert_window_rep_frame_ids.clear()
                    alert_window_rep_frames.clear()
                    alert_hit = False

            # Update windows only if not in cooldown
            if warn_cooldown_left <= 0:
                win_warning.append(1 if warn_hit else 0)
            if alert_cooldown_left <= 0:
                win_alert.append(1 if alert_hit else 0)

            warning_active = (len(win_warning) == int(WARNING_WINDOW_FRAMES) and sum(win_warning) >= int(
                WARNING_REQUIRE_HITS)) if warn_cooldown_left <= 0 else False
            alert_active_pre = (len(win_alert) == int(ALERT_WINDOW_FRAMES) and sum(win_alert) >= int(
                ALERT_REQUIRE_HITS)) if alert_cooldown_left <= 0 else False

            # Save alert-window frames when the pre-alert condition is first reached
            if bool(SAVE_ALERT_WINDOW_FRAMES) and alert_active_pre and not last_alert_pre_state:
                base = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
                save_dir = os.path.join(OUTPUT_ROOT, ALERT_WINDOW_SAVE_SUBDIR, f"{base}_{RUN_ID}")
                os.makedirs(save_dir, exist_ok=True)
                # Save exactly the current buffered window (up to ALERT_WINDOW_FRAMES inference frames)
                for (frm_bgr, box_rep, fid_rep) in list(alert_save_buf)[-int(ALERT_WINDOW_FRAMES):]:
                    out_img = frm_bgr.copy()
                    if box_rep is not None:
                        draw_box(out_img, box_rep, (0, 255, 0), 2, label=None, anchor="tl")
                    cv2.imwrite(os.path.join(save_dir, f"frame_{int(fid_rep):06d}.jpg"), out_img)

            last_alert_pre_state = bool(alert_active_pre)

            # Alert-window verify mode: convert pre-alert into alert or warning via extra cascaded ROI confirmation
            alert_active = False
            alert_window_verify_avg_conf_last = None
            if val_post_mode == "Alert-Window Cascade":
                # maintain a window buffer of representative boxes for potential cascaded ROI confirmation
                if rep_box is not None and alert_cooldown_left <= 0:
                    alert_window_rep_boxes.append(rep_box)
                    alert_window_rep_main_confs.append(float(rep_main_conf) if rep_main_conf is not None else 0.0)
                    alert_window_rep_frame_ids.append(int(frame_id))
                    alert_window_rep_frames.append(frame.copy())

                if alert_active_pre:
                    alert_window_verify_triggers += 1
                    # run cascaded ROI confirmation on the buffered window boxes (ROI around each rep box)
                    accepted_confs = []
                    verify_attempts_here = 0
                    verify_accepts_here = 0
                    verify_rejects_here = 0

                    t_verify0 = time.perf_counter()
                    for frm_win, b_win in zip(list(alert_window_rep_frames), list(alert_window_rep_boxes)):
                        cx, cy = center_of(b_win)
                        crop, roi = crop_square_around_point(frm_win, cx, cy, ROI_SIZE_LOCAL)
                        verify_rois.append(roi)
                        boxes_v, confs_v = infer_roi_and_project_to_frame(
                            model, DRONE_CLASS_ID, crop, (roi[0], roi[1], roi[2], roi[3]),
                            conf=ALERTWIN_CASCADE_DETECT_CONF, max_det=MAX_ROI_DETECTIONS
                        )
                        total_yolo_calls += 1
                        verify_attempts_here += 1
                        if boxes_v:
                            post_conf = float(confs_v[0]) if confs_v else 0.0
                            if post_conf >= float(ALERTWIN_CASCADE_ACCEPT_CONF):
                                accepted_confs.append(post_conf)
                                verify_accepts_here += 1
                            else:
                                verify_rejects_here += 1
                        else:
                            verify_rejects_here += 1

                    t_verify1 = time.perf_counter()
                    alertwin_cascade_time_s += (t_verify1 - t_verify0)

                    alert_window_verify_accepts += verify_accepts_here
                    alert_window_verify_rejects += verify_rejects_here

                    if accepted_confs:
                        alert_window_verify_avg_conf_last = float(np.mean(accepted_confs))
                    else:
                        alert_window_verify_avg_conf_last = 0.0

                    # Decide alert vs warning
                    if (verify_accepts_here >= int(ALERTWIN_CASCADE_MIN_ACCEPTS)) and (
                            alert_window_verify_avg_conf_last >= float(ALERTWIN_CASCADE_AVGCONF_ACCEPT)):
                        alert_active = True
                    else:
                        # If cascaded confirmation fails, treat this as a warning event.
                        alert_active = False
                        warning_active = True

                    # Persist a status line on the video log until the cascaded check + cooldown period elapses
                    verdict = "ALERT" if alert_active else "WARNING"
                    alertwin_cascade_status_msg = (
                        f"AlertWinCascade: pass={verify_accepts_here}/{int(ALERT_WINDOW_FRAMES)} "
                        f"avg={float(alert_window_verify_avg_conf_last):.2f} "
                        f"need>={float(ALERTWIN_CASCADE_AVGCONF_ACCEPT):.2f} "
                        f"minpass>={int(ALERTWIN_CASCADE_MIN_ACCEPTS)} => {verdict}"
                    )
                    # TTL in VIDEO frames: cooldown seconds after the check finishes
                    alertwin_cascade_status_ttl = max(alertwin_cascade_status_ttl, int(round(float(ALERT_COOLDOWN_S) * float(fps))))

                    # reset after event decision + start cooldown
                    if alert_active and not last_alert_state:
                        count_alert_events += 1
                        alert_cooldown_left = max(alert_cooldown_left, int(round(float(ALERT_COOLDOWN_S) * float(INFER_FPS))))
                    if warning_active and not last_warning_state:
                        count_warning_events += 1
                        warn_cooldown_left = max(warn_cooldown_left, int(round(float(WARNING_COOLDOWN_S) * float(INFER_FPS))))

                    temporal_alert.reset()
                    win_alert.clear()
                    alert_window_rep_boxes.clear()
                    alert_window_rep_main_confs.clear()
                    alert_window_rep_frame_ids.clear()
                    alert_window_rep_frames.clear()
            else:
                # normal counting
                alert_active = bool(alert_active_pre)

                if warning_active and not last_warning_state:
                    count_warning_events += 1
                    if warn_cooldown_frames > 0:
                        warn_cooldown_left = warn_cooldown_frames
                        temporal_warning.reset()
                        win_warning.clear()
                if alert_active and not last_alert_state:
                    count_alert_events += 1
                    if alert_cooldown_frames > 0:
                        alert_cooldown_left = alert_cooldown_frames
                        temporal_alert.reset()
                        win_alert.clear()

            last_warning_state = bool(warning_active)
            last_alert_state = bool(alert_active)

            # ============================
            # GT-based reference warning/alert (same windows + cooldown)
            # ============================
            gt_rep_box = gt_boxes[0] if gt_boxes else None
            gt_warn_hit = False
            gt_alert_hit = False

            if gt_warn_cooldown_left > 0:
                gt_warn_cooldown_left -= 1
                gt_temporal_warning.reset()
                gt_win_warning.clear()
            if gt_alert_cooldown_left > 0:
                gt_alert_cooldown_left -= 1
                gt_temporal_alert.reset()
                gt_win_alert.clear()

            if gt_rep_box is None:
                gt_temporal_warning.reset()
                gt_temporal_alert.reset()
                gt_win_warning.clear()
                gt_win_alert.clear()
                gt_warn_hit = False
                gt_alert_hit = False
            else:
                ok_gw, _, _, _ = gt_temporal_warning.accept(gt_rep_box, W, H)
                gt_warn_hit = bool(ok_gw)

                gt_area = box_area(gt_rep_box)
                if val_post_mode == "None":
                    size_ok_gt = (gt_area >= float(ALERT_MIN_AREA_PX2))
                else:
                    size_ok_gt = (gt_area >= float(ALERT_MIN_AREA_PX2)) or (gt_area >= float(ALERT_MIN_AREA_CASCADED_PX2))
                if size_ok_gt:
                    ok_ga, _, _, _ = gt_temporal_alert.accept(gt_rep_box, W, H)
                    gt_alert_hit = bool(ok_ga)
                else:
                    gt_temporal_alert.reset()
                    gt_win_alert.clear()
                    gt_alert_hit = False

            if gt_warn_cooldown_left <= 0:
                gt_win_warning.append(1 if gt_warn_hit else 0)
            if gt_alert_cooldown_left <= 0:
                gt_win_alert.append(1 if gt_alert_hit else 0)

            gt_warning_active = (len(gt_win_warning) == int(WARNING_WINDOW_FRAMES) and sum(gt_win_warning) >= int(
                WARNING_REQUIRE_HITS)) if gt_warn_cooldown_left <= 0 else False
            gt_alert_active = (len(gt_win_alert) == int(ALERT_WINDOW_FRAMES) and sum(gt_win_alert) >= int(
                ALERT_REQUIRE_HITS)) if gt_alert_cooldown_left <= 0 else False

            if gt_warning_active and not gt_last_warning_state:
                gt_count_warning_events += 1
                if warn_cooldown_frames > 0:
                    gt_warn_cooldown_left = warn_cooldown_frames
                    gt_temporal_warning.reset()
                    gt_win_warning.clear()
            if gt_alert_active and not gt_last_alert_state:
                gt_count_alert_events += 1
                if alert_cooldown_frames > 0:
                    gt_alert_cooldown_left = alert_cooldown_frames
                    gt_temporal_alert.reset()
                    gt_win_alert.clear()

            gt_last_warning_state = bool(gt_warning_active)
            gt_last_alert_state = bool(gt_alert_active)

            # Frame-level agreement (inference frames only)
            if warning_active and gt_warning_active:
                warn_state_tp += 1
            elif warning_active and not gt_warning_active:
                warn_state_fp += 1
            elif (not warning_active) and gt_warning_active:
                warn_state_fn += 1
            else:
                warn_state_tn += 1

            if alert_active and gt_alert_active:
                alert_state_tp += 1
            elif alert_active and not gt_alert_active:
                alert_state_fp += 1
            elif (not alert_active) and gt_alert_active:
                alert_state_fn += 1
            else:
                alert_state_tn += 1

            # Record inference row for metrics (FINAL detections only, after cascaded ROI confirmation filtering)
            inference_rows.append({
                "frame_id": int(frame_id),
                "gt_boxes": list(gt_boxes),
                "pred_boxes": list(verified_boxes[:int(MAX_FULLFRAME_DETECTIONS)]),
                "pred_confs": list(verified_confs[:int(MAX_FULLFRAME_DETECTIONS)]),
                "pred_sources": list(verified_sources[:int(MAX_FULLFRAME_DETECTIONS)]),
            })

            infer_t1 = time.perf_counter()
            total_infer_time_s += (infer_t1 - infer_t0)

            # Cache last inference state for non-infer frames
            last_infer_final_boxes = list(verified_boxes[:int(MAX_FULLFRAME_DETECTIONS)])
            last_infer_final_confs = list(verified_confs[:int(MAX_FULLFRAME_DETECTIONS)])
            last_infer_final_sources = list(verified_sources[:int(MAX_FULLFRAME_DETECTIONS)])
            last_infer_final_main_confs = list(verified_main_confs[:int(MAX_FULLFRAME_DETECTIONS)])
            last_infer_final_post_confs = list(verified_post_confs[:int(MAX_FULLFRAME_DETECTIONS)])

            last_infer_guided_rois = list(guided_rois)
            last_infer_verify_rois = list(verify_rois)

            last_infer_gate_prev_warn = warn_gate_prev
            last_infer_gate_curr_warn = warn_gate_curr
            last_infer_gate_prev_alert = alert_gate_prev
            last_infer_gate_curr_alert = alert_gate_curr
            last_infer_gate_warn_accepted = warn_acc
            last_infer_gate_alert_accepted = alert_acc

            last_infer_warn_state = bool(warning_active)
            last_infer_alert_state = bool(alert_active)
            last_infer_gt_warn_state = bool(gt_warning_active)
            last_infer_gt_alert_state = bool(gt_alert_active)

            last_infer_verify_stats = (verify_tried, verify_pass, verify_fail)
            last_infer_verify_details = list(verify_details)

        # ============================
        # Overlay & drawing (every frame, stable)
        # ============================

        if val_show_gt:
            for i, gb in enumerate(GT_DATA.get(frame_id, [])[:3]):
                if int(MIN_EVAL_AREA_PX2) > 0 and box_area(gb) < float(MIN_EVAL_AREA_PX2):
                    continue
                draw_box(vis, gb, (255, 255, 0), 2, label=f"GT{i + 1}", anchor="tr")

        # Draw predictions (final)
        for i, pb in enumerate(last_infer_final_boxes[:3]):
            src = last_infer_final_sources[i] if i < len(last_infer_final_sources) else "?"
            if src == "full":
                col = (0, 255, 0)
                tag = "FULL"
            elif src == "troi":
                col = (255, 0, 255)
                tag = "TROI"
            else:
                col = (0, 165, 255)
                tag = "CROI"
            label = f"P{i + 1}-{tag}" if SHOW_SOURCE_TAGS else None
            # alternate corners to avoid overlap
            anchors = ["tl", "tr", "bl"]
            draw_box(vis, pb, col, 1, label=label, anchor=anchors[i % len(anchors)])

        # Temporal ROI Propagations
        if val_show_guided:
            for roi in last_infer_guided_rois[:int(MAX_GUIDED_ROIS)]:
                draw_box(vis, roi, (255, 0, 255), 1, label="TemporalROI", anchor="bl")

        # Cascaded ROI Confirmation ROIs are log-only (do NOT draw boxes on video)
        # (verification passes are not part of the video frame stream)

        # Gates (prev/current)
        if val_show_gate:
            # if last_infer_gate_prev_warn is not None:
            #     draw_box(vis, last_infer_gate_prev_warn, (200, 200, 200), 1, label="WarnTCPrev", anchor="tl")
            # if last_infer_gate_curr_warn is not None:
            #     draw_box(vis, last_infer_gate_curr_warn, (255, 255, 255), 1, label="WarnTCCurr", anchor="tr")
            if last_infer_gate_prev_alert is not None:
                draw_box(vis, last_infer_gate_prev_alert, (0, 183, 235), 1, label="TCPrev", anchor="bl")
            if last_infer_gate_curr_alert is not None:
                draw_box(vis, last_infer_gate_curr_alert, (0, 255, 255), 1, label="TCCurr", anchor="br")

        # Top-left log
        if TOPLEFT_LOG_MODE != "off":
            if TOPLEFT_LOG_MODE == "windows_big":
                lines = [
                    f"WARNING hits {sum(win_warning)}/{len(win_warning)}  need {WARNING_REQUIRE_HITS}/{WARNING_WINDOW_FRAMES}  events={count_warning_events}",
                    f"ALERT   hits {sum(win_alert)}/{len(win_alert)}  need {ALERT_REQUIRE_HITS}/{ALERT_WINDOW_FRAMES}  events={count_alert_events}",
                ]
                if alertwin_cascade_status_msg:
                    lines.append(str(alertwin_cascade_status_msg))
                overlay_text_big(vis, lines)
            else:
                det_lines = []
                for i in range(min(3, len(last_infer_final_boxes))):
                    b = last_infer_final_boxes[i]
                    area = int(round(box_area(b)))
                    src = last_infer_final_sources[i] if i < len(last_infer_final_sources) else "?"
                    main_c = float(last_infer_final_main_confs[i]) if i < len(last_infer_final_main_confs) else float(last_infer_final_confs[i])
                    post_c = last_infer_final_post_confs[i] if i < len(last_infer_final_post_confs) else None
                    if src == "cascade" and post_c is not None:
                        det_lines.append(f"Pred{i + 1}: area={area} src=VER conf {main_c:.2f}->{float(post_c):.2f}")
                    else:
                        det_lines.append(f"Pred{i + 1}: area={area} src={src.upper()} conf {main_c:.2f}")

                det_line = " | ".join(det_lines) if det_lines else "Pred: none"

                vt, vp, vf = last_infer_verify_stats
                ver_line = f"Cascade: mode={CASCADED_ROI_CONFIRM_MODE} tried={vt} pass={vp} fail={vf}"
                if last_infer_verify_details:
                    # show last 3 attempt details so rejected attempts are visible
                    ver_line += " | " + " ; ".join(last_infer_verify_details[-3:])

                warn_gate_txt = "-" if last_infer_gate_warn_accepted is None else ("OK" if last_infer_gate_warn_accepted else "NO")
                alert_gate_txt = "-" if last_infer_gate_alert_accepted is None else ("OK" if last_infer_gate_alert_accepted else "NO")

                lines = [
                    f"Frame {frame_id} ({'INFER' if (frame_id % stride == 0) else 'HOLD'})  Pred={len(last_infer_final_boxes)}  UsedTROI={'YES' if (len(last_infer_guided_rois) > 0) else 'NO'}",
                    det_line,
                    ver_line,
                    f"Temporal Continuity: Warn={warn_gate_txt}  Alert={alert_gate_txt}",
                    f"Windows: Warn hits {sum(win_warning)}/{len(win_warning)} (need {WARNING_REQUIRE_HITS}/{WARNING_WINDOW_FRAMES})  |  Alert hits {sum(win_alert)}/{len(win_alert)} (need {ALERT_REQUIRE_HITS}/{ALERT_WINDOW_FRAMES})",
                    f"Events: Warning={count_warning_events}  Alert={count_alert_events}",
                ]
                if alertwin_cascade_status_msg:
                    lines.append(str(alertwin_cascade_status_msg))
                overlay_text(vis, lines)

        # Save video
        if writer is not None:
            writer.write(vis)

        # Real-time display (pause with PAUSE_KEY)
        if SHOW_WINDOW:
            show_img = vis
            if screen is not None:
                # ensure the full frame is visible by resizing to fitted display size
                show_img = cv2.resize(vis, (disp_w, disp_h), interpolation=cv2.INTER_AREA)

            cv2.imshow(WINDOW_NAME, show_img)
            k = cv2.waitKey(1) & 0xFF

            # Quit
            if k in (27, ord('q')):  # ESC or q
                break

            # Pause/resume (toggle)
            if k == ord(str(PAUSE_KEY).lower()):
                while True:
                    # keep showing the same frame while paused (window stays resizable)
                    cv2.imshow(WINDOW_NAME, show_img)
                    kk = cv2.waitKey(30) & 0xFF
                    if kk in (27, ord('q')):
                        ret = False  # force exit outer loop
                        break
                    if kk == ord(str(PAUSE_KEY).lower()):
                        break
                    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                        ret = False
                        break
                if not ret:
                    break

            # if window is closed by user
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break

        frame_id += 1

    cap.release()
    if writer is not None:
        writer.release()
    if SHOW_WINDOW:
        try:
            cv2.destroyWindow(WINDOW_NAME)
        except Exception:
            pass
    print("Done. Output video:", vid_out_path if vid_out_path else "(video disabled)")

    #            METRICS + SINGLE CSV

    mAP = compute_map50_greedy_iou50(inference_rows)

    tp_total = fp_total = fn_total = 0
    tp_confs = []
    fp_confs = []
    tp_confs_full = [];
    fp_confs_full = []
    tp_confs_guided = [];
    fp_confs_guided = []
    tp_confs_verify = [];
    fp_confs_verify = []

    for row in inference_rows:
        gts = row["gt_boxes"]
        preds = row["pred_boxes"]
        confs = row.get("pred_confs", [])
        srcs = row.get("pred_sources", ["?"] * len(preds))

        matched_gt = set()
        for pb, pc, ps in zip(preds, confs, srcs):
            best_iou = 0.0
            best_idx = -1
            for i, gb in enumerate(gts):
                if i in matched_gt:
                    continue
                v = iou_xyxy(pb, gb)
                if v > best_iou:
                    best_iou = v
                    best_idx = i
            if best_iou >= 0.5 and best_idx >= 0:
                tp_total += 1
                matched_gt.add(best_idx)
                tp_confs.append(float(pc))
                if ps == "full":
                    tp_confs_full.append(float(pc))
                elif ps == "troi":
                    tp_confs_guided.append(float(pc))
                elif ps == "cascade":
                    tp_confs_verify.append(float(pc))
            else:
                fp_total += 1
                fp_confs.append(float(pc))
                if ps == "full":
                    fp_confs_full.append(float(pc))
                elif ps == "troi":
                    fp_confs_guided.append(float(pc))
                elif ps == "cascade":
                    fp_confs_verify.append(float(pc))

        fn_total += (len(gts) - len(matched_gt))

    prec = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    rec = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    def safe_div(a, b):
        return float(a) / float(b) if b else 0.0

    warn_state_prec = safe_div(warn_state_tp, (warn_state_tp + warn_state_fp))
    warn_state_rec = safe_div(warn_state_tp, (warn_state_tp + warn_state_fn))
    warn_state_f1 = safe_div(2 * warn_state_prec * warn_state_rec, (warn_state_prec + warn_state_rec))

    alert_state_prec = safe_div(alert_state_tp, (alert_state_tp + alert_state_fp))
    alert_state_rec = safe_div(alert_state_tp, (alert_state_tp + alert_state_fn))
    alert_state_f1 = safe_div(2 * alert_state_prec * alert_state_rec, (alert_state_prec + alert_state_rec))

    def safe_mean(xs):
        return float(np.mean(xs)) if xs else 0.0

    all_pred_confs = [float(pc) for row in inference_rows for pc in row.get("pred_confs", [])]
    all_full_confs = [float(pc) for row in inference_rows for pc, ps in zip(row.get("pred_confs", []), row.get("pred_sources", [])) if ps == "full"]
    all_guided_confs = [float(pc) for row in inference_rows for pc, ps in zip(row.get("pred_confs", []), row.get("pred_sources", [])) if ps == "troi"]
    all_verify_confs = [float(pc) for row in inference_rows for pc, ps in zip(row.get("pred_confs", []), row.get("pred_sources", [])) if
                        ps == "cascade"]

    avg_infer_ms = (total_infer_time_s / num_infer_frames * 1000.0) if num_infer_frames else 0.0
    effective_infer_fps = (num_infer_frames / total_infer_time_s) if total_infer_time_s else 0.0
    effective_call_fps = (total_yolo_calls / total_infer_time_s) if total_infer_time_s else 0.0

    # CSV is append-only; a stable header is kept per output file.
    # We do NOT add new columns here; instead, we pack new config into run_id for traceability.
    run_id_tagged = (
        f"{RUN_ID}"
        f"|minEvalA={int(MIN_EVAL_AREA_PX2)}"
        f"|Wwin={int(WARNING_WINDOW_FRAMES)}:{int(WARNING_REQUIRE_HITS)}"
        f"|Awin={int(ALERT_WINDOW_FRAMES)}:{int(ALERT_REQUIRE_HITS)}"
        f"|Wcd={float(WARNING_COOLDOWN_S):.2f}"
        f"|Acd={float(ALERT_COOLDOWN_S):.2f}"
    )
    if val_post_mode == "Alert-Window Cascade":
        run_id_tagged += (
            f"|AWVconf={float(ALERTWIN_CASCADE_DETECT_CONF):.2f}"
            f"|AWVreq={float(ALERTWIN_CASCADE_ACCEPT_CONF):.2f}"
            f"|AWVavg={float(ALERTWIN_CASCADE_AVGCONF_ACCEPT):.2f}"
            f"|AWVmin={int(ALERTWIN_CASCADE_MIN_ACCEPTS)}"
        )

    detail = {
        "run_id": str(run_id_tagged),
        "video_name": os.path.basename(VIDEO_PATH),
        "output_video_name": os.path.basename(vid_out_path) if vid_out_path else "",

        "infer_fps_target": float(INFER_FPS),
        "video_fps": float(fps),
        "stride": int(stride),
        "imgsz": int(BASE_IMGSZ),
        "roi_size": int(ROI_SIZE_LOCAL),

        "temporal_roi_propagation_enabled": bool(val_pre),
        "troi_seed_min_area_px2": int(MIN_TROI_SEED_AREA_PX2),
        "cascaded_roi_confirmation_mode": str(val_post_mode),

        "detect_conf": float(DETECT_CONF),
        "troi_detect_conf": float(TROI_DETECT_CONF),
        "cascade_trigger_conf": float(CASCADE_TRIGGER_CONF),
        "cascade_detect_conf": float(CASCADE_DETECT_CONF),
        "cascade_accept_conf": float(CASCADE_ACCEPT_CONF),

        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "map50": float(mAP),

        "warning_events_pred": int(count_warning_events),
        "alert_events_pred": int(count_alert_events),

        "warning_events_gt": int(gt_count_warning_events),
        "alert_events_gt": int(gt_count_alert_events),

        "warning_state_tp": int(warn_state_tp),
        "warning_state_fp": int(warn_state_fp),
        "warning_state_fn": int(warn_state_fn),
        "warning_state_tn": int(warn_state_tn),
        "warning_state_precision": float(warn_state_prec),
        "warning_state_recall": float(warn_state_rec),
        "warning_state_f1": float(warn_state_f1),

        "alert_state_tp": int(alert_state_tp),
        "alert_state_fp": int(alert_state_fp),
        "alert_state_fn": int(alert_state_fn),
        "alert_state_tn": int(alert_state_tn),
        "alert_state_precision": float(alert_state_prec),
        "alert_state_recall": float(alert_state_rec),
        "alert_state_f1": float(alert_state_f1),

        "avg_conf_all_preds": safe_mean(all_pred_confs),
        "avg_conf_tp": safe_mean(tp_confs),
        "avg_conf_fp": safe_mean(fp_confs),

        "avg_conf_full_preds": safe_mean(all_full_confs),
        "avg_conf_guided_preds": safe_mean(all_guided_confs),
        "avg_conf_cascade_preds": safe_mean(all_verify_confs),

        # Cascaded ROI Confirmation summary:
        "cascade_attempts_total": int(len(conf_verify_trigger_main_all)),
        "cascade_pass_total": int(cnt_verify_pass),
        "cascade_fail_total": int(cnt_verify_fail),
        "avg_main_conf_cascade_attempts": safe_mean(conf_verify_trigger_main_all),
        "avg_post_conf_cascade_attempts_with_box": safe_mean(conf_verify_trigger_post_any),
        "avg_main_conf_cascade_pass": safe_mean(conf_verify_pass_main),
        "avg_post_conf_cascade_pass": safe_mean(conf_verify_pass_post),
        "avg_main_conf_cascade_fail": safe_mean(conf_verify_fail_main),
        "avg_post_conf_cascade_fail_with_box": safe_mean(conf_verify_fail_post_any),

        "avg_conf_tp_full": safe_mean(tp_confs_full),
        "avg_conf_fp_full": safe_mean(fp_confs_full),
        "avg_conf_tp_guided": safe_mean(tp_confs_guided),
        "avg_conf_fp_guided": safe_mean(fp_confs_guided),
        "avg_conf_tp_cascade": safe_mean(tp_confs_verify),
        "avg_conf_fp_cascade": safe_mean(fp_confs_verify),

        "num_infer_frames": int(len(inference_rows)),
        "num_gt_boxes": int(sum(len(r["gt_boxes"]) for r in inference_rows)),
        "num_pred_boxes": int(sum(len(r["pred_boxes"]) for r in inference_rows)),

        "total_inference_time_s": float(total_infer_time_s),
        "alertwin_cascade_time_s": float(alertwin_cascade_time_s),
        "avg_inference_time_ms": float(avg_infer_ms),
        "total_yolo_calls": int(total_yolo_calls),
        "effective_infer_fps": float(effective_infer_fps),
        "effective_call_fps": float(effective_call_fps),

        "model_weights": str(MODEL_WEIGHTS),
        "video_path": str(VIDEO_PATH),
        "annotation_path": str(ANNOTATION_PATH),
    }

    detail_csv_path = os.path.join(OUTPUT_ROOT, "demo_runs_detailed.csv")

    # Append-only: keep existing header if present; write only columns that already exist.
    if os.path.exists(detail_csv_path):
        with open(detail_csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
        cols = header if header else list(detail.keys())
    else:
        cols = list(detail.keys())

    write_head = not os.path.exists(detail_csv_path)
    with open(detail_csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_head:
            w.writerow(cols)
        w.writerow([detail.get(k, "") for k in cols])

    print("Saved single detailed summary CSV:", detail_csv_path)
    print(f"Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, mAP50: {mAP:.4f}")


if __name__ == "__main__":
    main()
