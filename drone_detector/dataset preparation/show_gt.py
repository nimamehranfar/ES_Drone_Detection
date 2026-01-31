import os
import cv2
import yaml

# ---- CONFIG ----
YAML_PATH = r"..\datasets\drone_mixed.yaml"
SPLIT = "test"                 # train | val | test
FILENAME_PREFIX = "mav_vid"
MAX_IMAGES = 50
# ----------------

with open(YAML_PATH, "r") as f:
    data = yaml.safe_load(f)

root = data["path"]
img_dir = os.path.join(root, data[SPLIT])
lbl_dir = img_dir.replace("images", "labels")

class_names = data["names"]

images = [
    f for f in os.listdir(img_dir)
    if f.lower().endswith((".jpg", ".png")) and f.startswith(FILENAME_PREFIX)
]

for img_name in images[:MAX_IMAGES]:
    img_path = os.path.join(img_dir, img_name)
    lbl_path = os.path.join(lbl_dir, os.path.splitext(img_name)[0] + ".txt")

    img = cv2.imread(img_path)
    if img is None:
        continue

    h, w = img.shape[:2]

    if os.path.exists(lbl_path):
        with open(lbl_path) as f:
            for line in f:
                cls, x, y, bw, bh = map(float, line.split())

                x1 = int((x - bw / 2) * w)
                y1 = int((y - bh / 2) * h)
                x2 = int((x + bw / 2) * w)
                y2 = int((y + bh / 2) * h)

                name = class_names[int(cls)]

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    img, name,
                    (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                )

    cv2.imshow("GT YOLO - mav-vid", img)
    if cv2.waitKey(0) == 27:  # ESC
        break

cv2.destroyAllWindows()
