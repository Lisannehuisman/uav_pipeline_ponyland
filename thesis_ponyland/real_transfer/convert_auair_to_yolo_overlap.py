from pathlib import Path
from PIL import Image
from collections import defaultdict, Counter
import json
import os
import shutil
import csv

SRC = Path("/vol/tensusers6/lisannehuisman/data/real_auair_raw")
IMG_ROOT = SRC / "images"
ANN_PATH = SRC / "annotations.json"

OUT = Path("/vol/tensusers6/lisannehuisman/projects/real_transfer/auair_overlap")
IMG_OUT = OUT / "images" / "test"
LAB_OUT = OUT / "labels" / "test"
IMG_OUT.mkdir(parents=True, exist_ok=True)
LAB_OUT.mkdir(parents=True, exist_ok=True)

# AU-AIR numeric class IDs, based on the AU-AIR class order:
# 0 = human, 1 = car, 2 = truck, 3 = van,
# 4 = motorbike, 5 = bike/bicycle, 6 = bus, 7 = trailer.
#
# Synthetic class IDs:
# 4 = whitevan
# 5 = suv
# 6 = male
AU_TO_SYN = {
    "0": 6,  # human -> male
    "1": 5,  # car -> suv
    "2": 4,  # truck -> whitevan
    "3": 4,  # van -> whitevan

    # Also support textual names in case some annotations use strings.
    "human": 6,
    "person": 6,
    "pedestrian": 6,
    "car": 5,
    "truck": 4,
    "van": 4,
}

SYN_NAMES = {
    0: "tent",
    1: "tank",
    2: "tower",
    3: "container",
    4: "whitevan",
    5: "suv",
    6: "male",
    7: "rock",
    8: "barrel",
    9: "tree",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

def norm_name(x):
    if x is None:
        return None
    return str(x).strip().lower().replace(" ", "").replace("_", "").replace("-", "")

def index_images():
    images = sorted([p for p in IMG_ROOT.rglob("*") if p.suffix.lower() in IMAGE_EXTS])
    if not images:
        raise RuntimeError(f"No images found under {IMG_ROOT}")

    index = {}
    for p in images:
        rel = str(p.relative_to(IMG_ROOT)).replace("\\", "/")
        index[p.name] = p
        index[rel] = p
    return images, index

def get_records(data):
    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ["annotations", "images", "frames", "samples", "data"]:
            if key in data and isinstance(data[key], list):
                return data[key]

    raise RuntimeError(f"Could not find frame records. Top-level keys: {list(data.keys())[:20] if isinstance(data, dict) else 'not a dict'}")

def get_image_name(record):
    for key in ["image_name", "file_name", "filename", "image", "name", "path", "img_path", "frame"]:
        if key in record and isinstance(record[key], str):
            return record[key]
    return None

def get_boxes(record):
    for key in ["bbox", "bboxes", "boxes", "objects", "annotations", "labels"]:
        if key in record and isinstance(record[key], list):
            return record[key]
    return []

def get_class_name(box):
    if isinstance(box, dict):
        for key in ["class", "category", "name", "label", "class_name", "category_name", "class_id", "category_id"]:
            if key in box:
                return box[key]
    return None

def get_bbox_xywh(box):
    if isinstance(box, dict):
        if all(k in box for k in ["left", "top", "width", "height"]):
            return float(box["left"]), float(box["top"]), float(box["width"]), float(box["height"])

        if all(k in box for k in ["x", "y", "w", "h"]):
            return float(box["x"]), float(box["y"]), float(box["w"]), float(box["h"])

        if "bbox" in box and isinstance(box["bbox"], (list, tuple)) and len(box["bbox"]) >= 4:
            return tuple(map(float, box["bbox"][:4]))

        if all(k in box for k in ["xmin", "ymin", "xmax", "ymax"]):
            x1 = float(box["xmin"])
            y1 = float(box["ymin"])
            x2 = float(box["xmax"])
            y2 = float(box["ymax"])
            return x1, y1, x2 - x1, y2 - y1

    if isinstance(box, (list, tuple)) and len(box) >= 4:
        return tuple(map(float, box[:4]))

    return None

def safe_out_name(img_path):
    rel = img_path.relative_to(IMG_ROOT)
    return "__".join(rel.parts)

if not IMG_ROOT.exists():
    raise RuntimeError(f"Image folder not found: {IMG_ROOT}")

if not ANN_PATH.exists():
    raise RuntimeError(f"Annotation file not found: {ANN_PATH}")

all_images, image_index = index_images()

with open(ANN_PATH, "r", encoding="utf-8", errors="ignore") as f:
    data = json.load(f)

records = get_records(data)

print("Images found:", len(all_images))
print("Frame records found:", len(records))

labels_by_image = defaultdict(list)
raw_counts = Counter()
mapped_counts = Counter()
ignored_or_unmapped = 0
unmatched_records = 0
bad_boxes = 0
linked_images = set()

for rec in records:
    if not isinstance(rec, dict):
        continue

    img_name = get_image_name(rec)

    if img_name is None:
        unmatched_records += 1
        continue

    img_name_clean = img_name.replace("\\", "/")
    img_path = image_index.get(img_name_clean) or image_index.get(Path(img_name_clean).name)

    if img_path is None:
        unmatched_records += 1
        continue

    try:
        with Image.open(img_path) as im:
            W, H = im.size
    except Exception:
        unmatched_records += 1
        continue

    out_name = safe_out_name(img_path)
    out_img_path = IMG_OUT / out_name
    out_lab_path = LAB_OUT / (Path(out_name).stem + ".txt")

    if not out_img_path.exists():
        try:
            os.symlink(img_path, out_img_path)
        except Exception:
            shutil.copy2(img_path, out_img_path)

    linked_images.add(out_img_path)

    boxes = get_boxes(rec)

    for box in boxes:
        cname = norm_name(get_class_name(box))
        raw_counts[cname or "unknown"] += 1

        if cname not in AU_TO_SYN:
            ignored_or_unmapped += 1
            continue

        bbox = get_bbox_xywh(box)

        if bbox is None:
            bad_boxes += 1
            continue

        x, y, w, h = bbox

        if w <= 0 or h <= 0:
            bad_boxes += 1
            continue

        xc = (x + w / 2.0) / W
        yc = (y + h / 2.0) / H
        wn = w / W
        hn = h / H

        xc = min(max(xc, 0.0), 1.0)
        yc = min(max(yc, 0.0), 1.0)
        wn = min(max(wn, 0.0), 1.0)
        hn = min(max(hn, 0.0), 1.0)

        syn_cls = AU_TO_SYN[cname]
        labels_by_image[out_lab_path].append((syn_cls, xc, yc, wn, hn))
        mapped_counts[syn_cls] += 1

for out_img_path in linked_images:
    lab = LAB_OUT / (out_img_path.stem + ".txt")
    if lab not in labels_by_image:
        labels_by_image[lab] = []

for lab_path, rows in labels_by_image.items():
    with open(lab_path, "w") as f:
        for cls, xc, yc, w, h in rows:
            f.write(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

yaml_path = OUT / "auair_overlap_synthetic_ids.yaml"
with open(yaml_path, "w") as f:
    f.write(f"path: {OUT}\n")
    f.write("train: images/test\n")
    f.write("val: images/test\n")
    f.write("test: images/test\n\n")
    f.write("names:\n")
    for i in range(10):
        f.write(f"  {i}: {SYN_NAMES[i]}\n")

summary_path = OUT / "conversion_summary.csv"
with open(summary_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["type", "class_id_or_name", "class_name", "count"])

    for k, v in sorted(raw_counts.items(), key=lambda x: str(x[0])):
        writer.writerow(["raw_auair", k, "", v])

    for k, v in sorted(mapped_counts.items()):
        writer.writerow(["mapped_synthetic", k, SYN_NAMES[k], v])

    writer.writerow(["info", "n_image_files_found", "", len(all_images)])
    writer.writerow(["info", "n_frame_records", "", len(records)])
    writer.writerow(["info", "n_images_linked", "", len(linked_images)])
    writer.writerow(["info", "n_label_files_written", "", len(labels_by_image)])
    writer.writerow(["info", "ignored_or_unmapped_boxes", "", ignored_or_unmapped])
    writer.writerow(["info", "unmatched_records", "", unmatched_records])
    writer.writerow(["info", "bad_boxes", "", bad_boxes])

print("\nDone.")
print("YOLO dataset:", OUT)
print("YAML:", yaml_path)
print("Summary:", summary_path)
print("Raw AU-AIR counts:", dict(raw_counts))
print("Mapped synthetic counts:", dict(mapped_counts))
print("Ignored/unmapped boxes:", ignored_or_unmapped)
print("Unmatched records:", unmatched_records)
print("Bad boxes:", bad_boxes)
