import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from tqdm import tqdm


IN_IMG_DIR = Path(r"E:\Dataset\DUT Anti-UAV Detection and Tracking\train\img")
IN_XML_DIR = Path(r"E:\Dataset\DUT Anti-UAV Detection and Tracking\train\xml")

OUT_ROOT = Path(r"E:\Dataset\YOLOv11_ready_rgb\dut_anti_uav_det")
OUT_IMG_DIR = OUT_ROOT / "images" / "train"
OUT_LBL_DIR = OUT_ROOT / "labels" / "train"


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def voc_to_yolo(xml_path: Path) -> tuple[str, int, int, list[tuple[int, float, float, float, float]]]:
    # returns: filename, width, height, list of (cls, cx, cy, w, h)
    root = ET.parse(str(xml_path)).getroot()
    filename = root.findtext("filename", default=xml_path.stem + ".jpg")

    size = root.find("size")
    if size is None:
        raise RuntimeError(f"Missing <size> in {xml_path}")
    w = int(size.findtext("width"))
    h = int(size.findtext("height"))

    yolo_objs = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip().lower()
        # DUT uses "UAV" in your sample
        cls = 0  # drone

        bnd = obj.find("bndbox")
        if bnd is None:
            continue

        xmin = float(bnd.findtext("xmin"))
        ymin = float(bnd.findtext("ymin"))
        xmax = float(bnd.findtext("xmax"))
        ymax = float(bnd.findtext("ymax"))

        bw = max(0.0, xmax - xmin)
        bh = max(0.0, ymax - ymin)
        if bw <= 0 or bh <= 0:
            continue

        cx = (xmin + xmax) / 2.0 / w
        cy = (ymin + ymax) / 2.0 / h
        nw = bw / w
        nh = bh / h

        cx = min(max(cx, 0.0), 1.0)
        cy = min(max(cy, 0.0), 1.0)
        nw = min(max(nw, 0.0), 1.0)
        nh = min(max(nh, 0.0), 1.0)

        yolo_objs.append((cls, cx, cy, nw, nh))

    return filename, w, h, yolo_objs


def main() -> None:
    ap = argparse.ArgumentParser()
    args = ap.parse_args()

    ensure_dir(OUT_IMG_DIR)
    ensure_dir(OUT_LBL_DIR)

    xml_files = sorted(IN_XML_DIR.glob("*.xml"))
    for xml_path in tqdm(xml_files, desc="DUT VOC->YOLO", unit="file"):
        filename, w, h, objs = voc_to_yolo(xml_path)

        src_img = IN_IMG_DIR / filename
        if not src_img.exists():
            # try stem match
            jpg = IN_IMG_DIR / (xml_path.stem + ".jpg")
            if jpg.exists():
                src_img = jpg
            else:
                continue

        dst_img = OUT_IMG_DIR / src_img.name
        shutil.copy2(src_img, dst_img)

        dst_lbl = OUT_LBL_DIR / (src_img.stem + ".txt")
        if objs:
            lines = [f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}" for cls, cx, cy, nw, nh in objs]
            dst_lbl.write_text("\n".join(lines), encoding="utf-8")
        else:
            dst_lbl.write_text("", encoding="utf-8")

    print(f"\nDONE. Output: {OUT_ROOT}")


if __name__ == "__main__":
    main()
