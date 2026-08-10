from pathlib import Path
from PIL import Image
from collections import defaultdict, Counter
import os
import shutil
import csv
import re

SRC = Path("/vol/tensusers6/lisannehuisman/data/real_testset_realimage")
IMG_ROOT = SRC / "uav_data"
GLOBAL_ANN = SRC / "uav_sequences.txt"

OUT = Path("/vol/tensusers6/lisannehuisman/projects/real_transfer/visdrone_mot_overlap")
IMG_OUT = OUT / "images" / "test"
LAB_OUT = OUT / "labels" / "test"
IMG_OUT.mkdir(parents=True, exist_ok=True)
LAB_OUT.mkdir(parents=True, exist_ok=True)

# VisDrone MOT category -> your synthetic YOLO class ID
# VisDrone: 1 pedestrian, 2 people, 4 car, 5 van
# Synthetic: 6 male, 5 suv, 4 whitevan
VIS_TO_SYN = {
    1: 6,  # pedestrian -> male
    2: 6,  # people -> male
    4: 5,  # car -> suv
    5: 4,  # van -> whitevan
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

def infer_sequence_name(img_path: Path):
    rel = img_path.relative_to(IMG_ROOT)
    parts = rel.parts

    # Common VisDrone-like structures:
    # uav_data/sequences/uav000.../0000001.jpg
    # uav_data/uav000.../0000001.jpg
    # uav_data/uav000.../img1/0000001.jpg
    if len(parts) >= 3 and parts[0].lower() in {"sequences", "images"}:
        return parts[1]
    if len(parts) >= 3 and parts[-2].lower() in {"img1", "images"}:
        return parts[-3]
    if len(parts) >= 2:
        return parts[0]
    return "global"

def out_image_name(img_path: Path):
    rel = img_path.relative_to(IMG_ROOT)
    safe = "__".join(rel.with_suffix("").parts)
    return safe + img_path.suffix.lower()

def parse_line(line):
    line = line.strip()
    if not line:
        return None
    parts = [p for p in re.split(r"[,\s]+", line) if p]

    # Supports either:
    # 10 columns: frame,id,x,y,w,h,score,cat,trunc,occ
    # 11+ columns: sequence,frame,id,x,y,w,h,score,cat,trunc,occ
    if len(parts) >= 11 and not parts[0].lstrip("-").replace(".", "", 1).isdigit():
        seq = parts[0]
        nums = parts[1:]
    elif len(parts) >= 10:
        seq = None
        nums = parts
    else:
        return None

    try:
        frame = int(float(nums[0]))
        target_id = int(float(nums[1]))
        x = float(nums[2])
        y = float(nums[3])
        w = float(nums[4])
        h = float(nums[5])
        score = float(nums[6])
        cat = int(float(nums[7]))
        trunc = int(float(nums[8])) if len(nums) > 8 else -1
        occ = int(float(nums[9])) if len(nums) > 9 else -1
    except Exception:
        return None

    return seq, frame, target_id, x, y, w, h, score, cat, trunc, occ

# 1. Index images
images = sorted([p for p in IMG_ROOT.rglob("*") if p.suffix.lower() in IMAGE_EXTS])
if not images:
    raise RuntimeError(f"No images found under {IMG_ROOT}")

image_index = {}
seqs = set()
for img in images:
    seq = infer_sequence_name(img)
    frame = frame_number_from_name(img)
    if frame is None:
        continue
    seqs.add(seq)

    out_name = out_image_name(img)
    image_index[(seq, frame)] = {
        "src": img,
        "out_name": out_name,
        "out_path": IMG_OUT / out_name,
        "label_path": LAB_OUT / (Path(out_name).stem + ".txt"),
    }

print(f"Found {len(image_index)} indexed images across {len(seqs)} sequence(s).")
print("Example sequence names:", sorted(list(seqs))[:10])

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

# 3. Find annotation files
ann_files = []
standard_ann_dirs = list(SRC.rglob("annotations"))
for d in standard_ann_dirs:
    ann_files.extend(sorted(d.glob("*.txt")))

# Also support your current single annotation file
if GLOBAL_ANN.exists():
    ann_files.append(GLOBAL_ANN)

ann_files = list(dict.fromkeys(ann_files))
if not ann_files:
    raise RuntimeError("No annotation txt files found. Expected annotations/*.txt or uav_sequences.txt")

print("Annotation files:")
for a in ann_files[:20]:
    print(" ", a)
if len(ann_files) > 20:
    print(f" ... and {len(ann_files)-20} more")

labels_by_image = defaultdict(list)
raw_vis_counts = Counter()
mapped_counts = Counter()
unmatched = 0
ignored = 0

single_sequence = None
if len(seqs) == 1:
    single_sequence = next(iter(seqs))

for ann in ann_files:
    default_seq = ann.stem

    # If the global file has a generic name and there is exactly one image sequence,
    # use that one sequence. If lines include a sequence name, that overrides this.
    if ann.name == "uav_sequences.txt" and single_sequence is not None:
        default_seq = single_sequence

    with open(ann, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parsed = parse_line(line)
            if parsed is None:
                continue

            seq, frame, target_id, x, y, w, h, score, cat, trunc, occ = parsed
            if seq is None:
                seq = default_seq

            raw_vis_counts[cat] += 1

            # score 0 or category 0 are ignored regions in VisDrone
            if score == 0 or cat == 0:
                ignored += 1
                continue

            if cat not in VIS_TO_SYN:
                ignored += 1
                continue

            key = (seq, frame)

            # If global labels do not contain sequence names and several sequences exist,
            # we cannot safely know which sequence the frame belongs to.
            if key not in image_index and ann.name == "uav_sequences.txt" and len(seqs) > 1:
                # Try matching by frame only if it is unique across all sequences
                candidates = [k for k in image_index if k[1] == frame]
                if len(candidates) == 1:
                    key = candidates[0]

            if key not in image_index:
                unmatched += 1
                continue

            img_path = image_index[key]["src"]
            with Image.open(img_path) as im:
                W, H = im.size

            # Convert pixel xywh top-left to normalized YOLO xywh center
            xc = (x + w / 2.0) / W
            yc = (y + h / 2.0) / H
            wn = w / W
            hn = h / H

            # Clip lightly to avoid invalid labels
            xc = min(max(xc, 0.0), 1.0)
            yc = min(max(yc, 0.0), 1.0)
            wn = min(max(wn, 0.0), 1.0)
            hn = min(max(hn, 0.0), 1.0)

            syn_cls = VIS_TO_SYN[cat]
            labels_by_image[key].append((syn_cls, xc, yc, wn, hn))
            mapped_counts[syn_cls] += 1

# 4. Write YOLO label files. Empty files are okay for frames without mapped overlap objects.
for key, meta in image_index.items():
    label_path = meta["label_path"]
    rows = labels_by_image.get(key, [])
    with open(label_path, "w") as f:
        for cls, xc, yc, w, h in rows:
            f.write(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

# 5. Write YAML
yaml_path = OUT / "visdrone_mot_overlap_synthetic_ids.yaml"
with open(yaml_path, "w") as f:
    f.write(f"path: {OUT}\n")
    f.write("train: images/test\n")
    f.write("val: images/test\n")
    f.write("test: images/test\n\n")
    f.write("names:\n")
    for i in range(10):
        f.write(f"  {i}: {SYN_NAMES[i]}\n")

# 6. Write summary
summary_path = OUT / "conversion_summary.csv"
with open(summary_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["type", "class_id", "class_name", "count"])
    for k, v in sorted(raw_vis_counts.items()):
        writer.writerow(["raw_visdrone", k, "", v])
    for k, v in sorted(mapped_counts.items()):
        writer.writerow(["mapped_synthetic", k, SYN_NAMES[k], v])
    writer.writerow(["info", "n_images", "", len(image_index)])
    writer.writerow(["info", "n_sequences", "", len(seqs)])
    writer.writerow(["info", "ignored_or_unmapped_boxes", "", ignored])
    writer.writerow(["info", "unmatched_boxes", "", unmatched])

print("\nDone.")
print("YOLO dataset:", OUT)
print("YAML:", yaml_path)
print("Summary:", summary_path)
print("Mapped synthetic counts:", dict(mapped_counts))
print("Ignored/unmapped boxes:", ignored)
print("Unmatched boxes:", unmatched)

if unmatched > 0:
    print("\nWARNING: Some annotation rows could not be matched to images.")
    print("If this is high, inspect the structure and the first lines of uav_sequences.txt.")
