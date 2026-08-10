from pathlib import Path
from collections import Counter
import os
import shutil
import csv
import yaml

SRC = Path("/vol/tensusers6/lisannehuisman/data/real_new_dataset_raw")
SRC_YAML = SRC / "data.yaml"

OUT = Path("/vol/tensusers6/lisannehuisman/projects/real_transfer/new_real_dataset_overlap")
IMG_OUT = OUT / "images" / "test"
LAB_OUT = OUT / "labels" / "test"

IMG_OUT.mkdir(parents=True, exist_ok=True)
LAB_OUT.mkdir(parents=True, exist_ok=True)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

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

# Map possible real-dataset class names to your synthetic model class IDs.
# Add or edit synonyms here if your data.yaml uses different names.
NAME_TO_SYN_ID = {
    "tent": 0,
    "tank": 1,
    "tower": 2,
    "container": 3,
    "shippingcontainer": 3,
    "whitevan": 4,
    "van": 4,
    "truck": 4,
    "white-vans": 4,
    "white_van": 4,
    "suv": 5,
    "car": 5,
    "vehicle": 5,
    "male": 6,
    "person": 6,
    "human": 6,
    "pedestrian": 6,
    "rock": 7,
    "barrel": 8,
    "tree": 9,
}

def norm_name(x):
    return str(x).strip().lower().replace(" ", "").replace("_", "").replace("-", "")

def read_yaml_names(yaml_path):
    with open(yaml_path, "r", encoding="utf-8", errors="ignore") as f:
        data = yaml.safe_load(f)

    names = data.get("names")
    if names is None:
        raise RuntimeError("No 'names' field found in data.yaml")

    if isinstance(names, dict):
        # keys may be strings or ints
        id_to_name = {int(k): str(v) for k, v in names.items()}
    elif isinstance(names, list):
        id_to_name = {i: str(v) for i, v in enumerate(names)}
    else:
        raise RuntimeError(f"Unsupported names format: {type(names)}")

    return data, id_to_name

def resolve_split_path(data, split_name):
    val = data.get(split_name)
    if val is None:
        return None

    if isinstance(val, list):
        return [resolve_one_path(v) for v in val]

    return [resolve_one_path(val)]

def resolve_one_path(p):
    p = Path(str(p))

    # Absolute path
    if p.is_absolute():
        return p

    # Relative to YAML path field if present, otherwise relative to SRC
    return (SRC / p).resolve()

def collect_images_from_paths(paths):
    imgs = []
    for p in paths:
        if p.is_file() and p.suffix.lower() == ".txt":
            with open(p, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        lp = Path(line)
                        if not lp.is_absolute():
                            lp = (p.parent / lp).resolve()
                        imgs.append(lp)
        elif p.is_dir():
            imgs.extend(sorted([x for x in p.rglob("*") if x.suffix.lower() in IMAGE_EXTS]))
        elif p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            imgs.append(p)
    return sorted(list(dict.fromkeys(imgs)))

def find_label_for_image(img_path):
    # Common YOLO structure: images/test/name.jpg -> labels/test/name.txt
    parts = list(img_path.parts)
    candidates = []

    if "images" in parts:
        idx = parts.index("images")
        label_parts = parts.copy()
        label_parts[idx] = "labels"
        candidates.append(Path(*label_parts).with_suffix(".txt"))

    # Also try sibling labels folder near root
    candidates.append(SRC / "labels" / img_path.with_suffix(".txt").name)
    candidates.append(SRC / "labels" / "test" / img_path.with_suffix(".txt").name)
    candidates.append(SRC / "labels" / "val" / img_path.with_suffix(".txt").name)
    candidates.append(SRC / "labels" / "train" / img_path.with_suffix(".txt").name)

    for c in candidates:
        if c.exists():
            return c

    return None

if not SRC_YAML.exists():
    raise RuntimeError(f"Cannot find {SRC_YAML}")

data, old_id_to_name = read_yaml_names(SRC_YAML)

print("Original class mapping from data.yaml:")
for k, v in sorted(old_id_to_name.items()):
    print(f"  {k}: {v}")

old_to_new = {}
unmapped_classes = {}

for old_id, name in old_id_to_name.items():
    key = norm_name(name)
    if key in NAME_TO_SYN_ID:
        old_to_new[old_id] = NAME_TO_SYN_ID[key]
    else:
        unmapped_classes[old_id] = name

print("\nOld class ID -> synthetic model class ID:")
for old_id, new_id in sorted(old_to_new.items()):
    print(f"  {old_id} ({old_id_to_name[old_id]}) -> {new_id} ({SYN_NAMES[new_id]})")

if unmapped_classes:
    print("\nUnmapped classes will be ignored:")
    for old_id, name in unmapped_classes.items():
        print(f"  {old_id}: {name}")

# Prefer test, then val, then train, then all images under SRC.
split = None
split_paths = None

for candidate in ["test", "val", "train"]:
    split_paths = resolve_split_path(data, candidate)
    if split_paths:
        split = candidate
        break

if split_paths:
    images = collect_images_from_paths(split_paths)
else:
    split = "all_found"
    images = sorted([p for p in SRC.rglob("*") if p.suffix.lower() in IMAGE_EXTS])

if not images:
    raise RuntimeError("No images found.")

print(f"\nUsing split: {split}")
print(f"Images found for evaluation: {len(images)}")

raw_label_counts = Counter()
mapped_label_counts = Counter()
missing_labels = 0
empty_labels = 0
ignored_boxes = 0
bad_lines = 0
linked_images = 0
written_label_files = 0

for img_path in images:
    # Use a safe flat output name preserving enough path context
    try:
        rel = img_path.relative_to(SRC)
        out_name = "__".join(rel.parts)
    except Exception:
        out_name = img_path.name

    out_img = IMG_OUT / out_name
    out_lab = LAB_OUT / (Path(out_name).stem + ".txt")

    if not out_img.exists():
        try:
            os.symlink(img_path, out_img)
        except Exception:
            shutil.copy2(img_path, out_img)

    linked_images += 1

    lab_path = find_label_for_image(img_path)
    rows_out = []

    if lab_path is None:
        missing_labels += 1
    else:
        with open(lab_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = [line.strip() for line in f if line.strip()]

        if not lines:
            empty_labels += 1

        for line in lines:
            parts = line.split()
            if len(parts) < 5:
                bad_lines += 1
                continue

            try:
                old_cls = int(float(parts[0]))
                x, y, w, h = map(float, parts[1:5])
            except Exception:
                bad_lines += 1
                continue

            raw_label_counts[old_cls] += 1

            if old_cls not in old_to_new:
                ignored_boxes += 1
                continue

            new_cls = old_to_new[old_cls]

            # YOLO labels should already be normalized. Clip lightly.
            x = min(max(x, 0.0), 1.0)
            y = min(max(y, 0.0), 1.0)
            w = min(max(w, 0.0), 1.0)
            h = min(max(h, 0.0), 1.0)

            if w <= 0 or h <= 0:
                bad_lines += 1
                continue

            rows_out.append(f"{new_cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
            mapped_label_counts[new_cls] += 1

    with open(out_lab, "w") as f:
        for row in rows_out:
            f.write(row + "\n")

    written_label_files += 1

# Write evaluation YAML with your model's synthetic class names
yaml_out = OUT / "new_real_dataset_synthetic_ids.yaml"
with open(yaml_out, "w") as f:
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
    writer.writerow(["type", "class_id", "class_name", "count"])

    for old_cls, count in sorted(raw_label_counts.items()):
        writer.writerow(["raw_original", old_cls, old_id_to_name.get(old_cls, ""), count])

    for new_cls, count in sorted(mapped_label_counts.items()):
        writer.writerow(["mapped_synthetic", new_cls, SYN_NAMES[new_cls], count])

    writer.writerow(["info", "split_used", split, ""])
    writer.writerow(["info", "n_images", "", len(images)])
    writer.writerow(["info", "linked_images", "", linked_images])
    writer.writerow(["info", "written_label_files", "", written_label_files])
    writer.writerow(["info", "missing_labels", "", missing_labels])
    writer.writerow(["info", "empty_original_labels", "", empty_labels])
    writer.writerow(["info", "ignored_unmapped_boxes", "", ignored_boxes])
    writer.writerow(["info", "bad_lines", "", bad_lines])

print("\nDone.")
print("Prepared dataset:", OUT)
print("YAML:", yaml_out)
print("Summary:", summary_path)
print("Raw label counts:", dict(raw_label_counts))
print("Mapped label counts:", dict(mapped_label_counts))
print("Missing labels:", missing_labels)
print("Ignored unmapped boxes:", ignored_boxes)
print("Bad lines:", bad_lines)
