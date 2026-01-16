#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
from collections import Counter, defaultdict

try:
    import yaml
except ImportError:
    yaml = None


def load_nc_from_data_yaml(data_yaml: Path) -> int:
    if yaml is None:
        raise RuntimeError("pyyaml is required: pip install pyyaml")
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    if "nc" in data and isinstance(data["nc"], int):
        return data["nc"]
    if "names" in data and isinstance(data["names"], (list, dict)):
        return len(data["names"])
    raise RuntimeError(f"Could not infer nc from {data_yaml}. Expected 'nc' or 'names'.")


def parse_label_line(line: str):
    parts = line.strip().split()
    if len(parts) != 5:
        return None, f"Expected 5 columns, got {len(parts)}"
    c_str, x_str, y_str, w_str, h_str = parts
    try:
        c = int(float(c_str))  # tolerate "0.0"
    except Exception:
        return None, f"Class is not an integer: {c_str}"
    try:
        x = float(x_str); y = float(y_str); w = float(w_str); h = float(h_str)
    except Exception:
        return None, f"Box values not floats: {parts[1:]}"
    return (c, x, y, w, h), None


def main():
    ap = argparse.ArgumentParser(description="Scan YOLO .txt labels for anomalies.")
    ap.add_argument("--labels", default=r"E:\Dataset\Visible_mix_with_no_drone\labels", help="Path to labels folder (contains .txt files).")
    ap.add_argument("--data", default="..\datasets\drone_mixed.yaml", help="Path to dataset YAML (to read nc/names).")
    ap.add_argument("--out", default="label_anomaly_report.csv", help="Output CSV filename.")
    ap.add_argument("--min_boxes_outlier", type=int, default=200, help="Flag images with >= this many boxes.")
    args = ap.parse_args()

    labels_dir = Path(args.labels)
    data_yaml = Path(args.data)
    out_csv = Path(args.out)

    if not labels_dir.exists():
        raise SystemExit(f"Labels dir not found: {labels_dir}")
    if not data_yaml.exists():
        raise SystemExit(f"Data YAML not found: {data_yaml}")

    nc = load_nc_from_data_yaml(data_yaml)

    issues = []  # rows for CSV
    class_counts = Counter()
    per_file_counts = {}
    bad_files = 0

    txt_files = sorted(labels_dir.rglob("*.txt"))
    if not txt_files:
        raise SystemExit(f"No .txt labels found under: {labels_dir}")

    for f in txt_files:
        lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
        box_count = 0
        file_class_counts = Counter()

        for i, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            parsed, err = parse_label_line(line)
            if err:
                issues.append({
                    "file": str(f),
                    "line": i,
                    "type": "FORMAT",
                    "detail": err,
                    "raw": line.strip()
                })
                continue

            c, x, y, w, h = parsed
            box_count += 1
            file_class_counts[c] += 1
            class_counts[c] += 1

            # class range
            if c < 0 or c >= nc:
                issues.append({
                    "file": str(f),
                    "line": i,
                    "type": "CLASS_OUT_OF_RANGE",
                    "detail": f"class={c} but nc={nc}",
                    "raw": line.strip()
                })

            # sanity checks for normalized YOLO coords
            # (Ultralytics expects normalized [0,1] unless you purposely trained otherwise)
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                issues.append({
                    "file": str(f),
                    "line": i,
                    "type": "XY_OUT_OF_RANGE",
                    "detail": f"x,y out of [0,1]: ({x:.6f},{y:.6f})",
                    "raw": line.strip()
                })
            if w <= 0 or h <= 0:
                issues.append({
                    "file": str(f),
                    "line": i,
                    "type": "WH_NON_POSITIVE",
                    "detail": f"w,h must be >0: ({w:.6f},{h:.6f})",
                    "raw": line.strip()
                })
            if w > 1.0 or h > 1.0:
                issues.append({
                    "file": str(f),
                    "line": i,
                    "type": "WH_TOO_LARGE",
                    "detail": f"w,h > 1.0: ({w:.6f},{h:.6f})",
                    "raw": line.strip()
                })

            # optional: flag ultra-tiny boxes (common drone failure mode)
            if w * h < 1e-6:
                issues.append({
                    "file": str(f),
                    "line": i,
                    "type": "BOX_ULTRA_TINY",
                    "detail": f"area={w*h:.3e} (very tiny box)",
                    "raw": line.strip()
                })

        per_file_counts[str(f)] = box_count

        # flag images with extreme box counts
        if box_count >= args.min_boxes_outlier:
            issues.append({
                "file": str(f),
                "line": 0,
                "type": "BOX_COUNT_OUTLIER",
                "detail": f"{box_count} boxes in one label file",
                "raw": ""
            })

        # flag files with only out-of-range classes (often indicates wrong label set)
        if box_count > 0:
            in_range = sum(v for k, v in file_class_counts.items() if 0 <= k < nc)
            if in_range == 0:
                issues.append({
                    "file": str(f),
                    "line": 0,
                    "type": "NO_IN_RANGE_CLASSES",
                    "detail": f"{box_count} boxes but none with class in [0,{nc-1}]",
                    "raw": ""
                })

        if any(r["file"] == str(f) for r in issues):
            bad_files += 1

    # extra: report missing classes in dataset (within expected range)
    missing = [c for c in range(nc) if class_counts[c] == 0]
    for c in missing:
        issues.append({
            "file": "",
            "line": 0,
            "type": "CLASS_MISSING_IN_DATA",
            "detail": f"class {c} never appears in labels",
            "raw": ""
        })

    # write CSV
    fieldnames = ["file", "line", "type", "detail", "raw"]
    with out_csv.open("w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=fieldnames)
        w.writeheader()
        for row in issues:
            w.writerow(row)

    # console summary
    total_boxes = sum(per_file_counts.values())
    print(f"Scanned: {len(txt_files)} label files")
    print(f"Total boxes: {total_boxes}")
    print(f"Files with at least one issue flagged: {bad_files}")
    print(f"Issues written to: {out_csv.resolve()}")
    print("Top classes by count (raw IDs):", class_counts.most_common(10))


if __name__ == "__main__":
    main()
