from pathlib import Path
from PIL import Image
from collections import defaultdict, Counter
import os
import shutil
import csv
import re

SRC = Path("/vol/tensusers6/lisannehuisman/data/real_testset_uav342")
IMG_ROOT = SRC / "uav342"
ANN_FILE = SRC / "uav342.txt"

OUT = Path("/vol/tensusers6/lisannehuisman/projects/real_transfer/visdrone_uav342_overlap")
IMG_OUT = OUT / "images" / "test"
LAB_OUT = OUT / "labels" / "test"
IMG_OUT.mkdir(parents=True, exist_ok=True)
LAB_OUT.mkdir(parents=True, exist_ok=True)

# VisDrone MOT category -> your synthetic YOLO class ID
# VisDrone: 1 pedestrian, 2 people, 4 car, 5 van
# Synthetic: 6 male, 5 suv, 4 whitevan
VIS_TO_SYN = {
    1: 6,
    2: 6,
    4: 5,
    5: 4,
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

def frame_number_from_name(path: Path):
    nums = re.findall(r"\d+", path.stem)
    if not nums:
        return None
    return int(nums[-1])

def parse_line(line):
    line = line.strip()
    if not line:
        return None

    parts = [p for p in re.split(r"[,\s]+", line) if p]
    if len(parts) < 10:
        return None

    try:
        frame = int(float(parts[0]))
        target_id = int(float(parts[1]))
        x = float(parts[2])
        y = float(parts[3])
        w = float(parts[4])
        h = float(parts[5])
        score = float(parts[6])
        cat = int(float(parts[7]))
        trunc = int(float(parts[8]))
        occ = int(float(parts[9]))
    except Exception:
        return None

    return frame, target_id, x, y, w, h, score, cat, trunc, occ

# 1. Index images by frame number
images = sorted([p for p in IMG_ROOT.rglob("*") if p.suffix.lower() in IMAGE_EXTS])
if not images:
    raise RuntimeError(f"No images found under {IMG_ROOT}")

image_index = {}
for img in images:
    frame = frame_number_from_name(img)
    if frame is None:
        continue

    out_name = img.name
    image_index[frame] = {
        "src": img,
        "out_name": out_name,
        "out_path": IMG_OUT / out_name,
        "label_path": LAB_OUT / (Path(out_name).stem + ".txt"),
    }

print(f"Found {len(image_index)} indexed images.")
print("First images:", list(image_index.keys())[:5])

# 2. Symlink/copy images into YOLO dataset folder
for meta in image_index.values():
    src = meta["src"]
    dst = meta["out_path"]
    if dst.exists():
        continue
    try:
        os.symlink(src, dst)
    except Exception:
        shutil.copy2(src, dst)

if not ANN_FILE.exists():
    raise RuntimeError(f"Annotation file not found: {ANN_FILE}")

labels_by_image = defaultdict(list)
raw_vis_counts = Counter()
mapped_counts = Counter()
ignored = 0
unmatched = 0

with open(ANN_FILE, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        parsed = parse_line(line)
        if parsed is None:
            continue

        frame, target_id, x, y, w, h, score, cat, trunc, occ = parsed
        raw_vis_counts[cat] += 1

        # VisDrone category 0 / score 0 are ignored regions
        if score == 0 or cat == 0:
            ignored += 1
            continue

        if cat not in VIS_TO_SYN:
            ignored += 1
            continue

        if frame not in image_index:
            unmatched += 1
            continue

        img_path = image_index[frame]["src"]
        with Image.open(img_path) as im:
            W, H = im.size

        xc = (x + w / 2.0) / W
        yc = (y + h / 2.0) / H
        wn = w / W
        hn = h / H

        xc = min(max(xc, 0.0), 1.0)
        yc = min(max(yc, 0.0), 1.0)
        wn = min(max(wn, 0.0), 1.0)
        hn = min(max(hn, 0.0), 1.0)

        syn_cls = VIS_TO_SYN[cat]
        labels_by_image[frame].append((syn_cls, xc, yc, wn, hn))
        mapped_counts[syn_cls] += 1

# 3. Write label files
for frame, meta in image_index.items():
    label_path = meta["label_path"]
    rows = labels_by_image.get(frame, [])
    with open(label_path, "w") as f:
        for cls, xc, yc, w, h in rows:
            f.write(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

# 4. Write YAML
yaml_path = OUT / "visdrone_uav342_overlap_synthetic_ids.yaml"
with open(yaml_path, "w") as f:
    f.write(f"path: {OUT}\n")
    f.write("train: images/test\n")
    f.write("val: images/test\n")
    f.write("test: images/test\n\n")
    f.write("names:\n")
    for i in range(10):
        f.write(f"  {i}: {SYN_NAMES[i]}\n")

# 5. Write summary
summary_path = OUT / "conversion_summary.csv"
with open(summary_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["type", "class_id", "class_name", "count"])
    for k, v in sorted(raw_vis_counts.items()):
        writer.writerow(["raw_visdrone", k, "", v])
    for k, v in sorted(mapped_counts.items()):
        writer.writerow(["mapped_synthetic", k, SYN_NAMES[k], v])
    writer.writerow(["info", "n_images", "", len(image_index)])
    writer.writerow(["info", "ignored_or_unmapped_boxes", "", ignored])
    writer.writerow(["info", "unmatched_boxes", "", unmatched])

print("\nDone.")
print("YOLO dataset:", OUT)
print("YAML:", yaml_path)
print("Summary:", summary_path)
print("Mapped synthetic counts:", dict(mapped_counts))
print("Ignored/unmapped boxes:", ignored)
print("Unmatched boxes:", unmatched)
