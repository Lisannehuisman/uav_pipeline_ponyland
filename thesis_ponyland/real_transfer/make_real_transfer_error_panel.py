from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import csv
import math

DATASET_ROOT = Path("/vol/tensusers6/lisannehuisman/projects/real_transfer/visdrone_uav342_overlap")
PRED_RUN = Path("/vol/tensusers6/lisannehuisman/projects/real_transfer/runs/yolov8l_M4_visdrone_uav342_allclasses_conf010")

IMG_DIR = DATASET_ROOT / "images" / "test"
GT_DIR = DATASET_ROOT / "labels" / "test"
PRED_DIR = PRED_RUN / "labels"

OUT_DIR = Path("/vol/tensusers6/lisannehuisman/projects/real_transfer/qualitative_figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_IMAGE = OUT_DIR / "visdrone_uav342_error_panel.jpg"
OUT_CSV = OUT_DIR / "visdrone_uav342_error_examples.csv"

CLASS_NAMES = {
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
IOU_THRESHOLD = 0.5

def read_yolo_labels(path, is_pred=False):
    boxes = []
    if not path.exists():
        return boxes

    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            cls = int(float(parts[0]))
            x, y, w, h = map(float, parts[1:5])
            conf = float(parts[5]) if is_pred and len(parts) > 5 else None

            boxes.append({
                "cls": cls,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "conf": conf,
            })

    return boxes

def xywh_to_xyxy(box, W, H):
    xc = box["x"] * W
    yc = box["y"] * H
    w = box["w"] * W
    h = box["h"] * H

    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2

    return max(0, x1), max(0, y1), min(W, x2), min(H, y2)

def iou(box_a, box_b, W, H):
    ax1, ay1, ax2, ay2 = xywh_to_xyxy(box_a, W, H)
    bx1, by1, bx2, by2 = xywh_to_xyxy(box_b, W, H)

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - inter
    return inter / union if union > 0 else 0

def evaluate_image(img_path, gt_boxes, pred_boxes):
    with Image.open(img_path) as im:
        W, H = im.size

    matched_preds = set()
    fn = 0
    wrong = 0

    for gt in gt_boxes:
        best_iou = 0
        best_pi = None

        for pi, pred in enumerate(pred_boxes):
            score = iou(gt, pred, W, H)
            if score > best_iou:
                best_iou = score
                best_pi = pi

        if best_iou >= IOU_THRESHOLD and best_pi is not None:
            matched_preds.add(best_pi)
            if pred_boxes[best_pi]["cls"] != gt["cls"]:
                wrong += 1
        else:
            fn += 1

    fp = 0
    hallucinated = 0

    for pi, pred in enumerate(pred_boxes):
        if pi in matched_preds:
            continue

        fp += 1
        if pred["cls"] not in {4, 5, 6}:
            hallucinated += 1

    return {
        "image": img_path.name,
        "n_gt": len(gt_boxes),
        "n_pred": len(pred_boxes),
        "false_negatives": fn,
        "false_positives": fp,
        "wrong_class": wrong,
        "hallucinated_non_overlap_classes": hallucinated,
        "score": wrong * 20 + hallucinated * 5 + fp + fn,
    }

def draw_boxes(img_path, boxes, title, mode):
    im = Image.open(img_path).convert("RGB")
    W, H = im.size

    target_w = 640
    scale = target_w / W
    target_h = int(H * scale)
    im = im.resize((target_w, target_h))

    header_h = 54
    canvas = Image.new("RGB", (target_w, target_h + header_h), "white")
    canvas.paste(im, (0, header_h))

    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
        small_font = ImageFont.truetype("DejaVuSans.ttf", 13)
    except Exception:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    draw.text((8, 6), title, fill=(0, 0, 0), font=font)

    if mode == "gt":
        color = (0, 170, 0)
        prefix = "GT"
    else:
        color = (220, 0, 0)
        prefix = "Pred"

    for b in boxes:
        x1, y1, x2, y2 = xywh_to_xyxy(b, W, H)
        x1 *= scale
        x2 *= scale
        y1 = y1 * scale + header_h
        y2 = y2 * scale + header_h

        cls_name = CLASS_NAMES.get(b["cls"], str(b["cls"]))

        if mode == "pred" and b["conf"] is not None:
            label = f"{prefix}: {cls_name} {b['conf']:.2f}"
        else:
            label = f"{prefix}: {cls_name}"

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        text_y = max(header_h, y1 - 18)
        draw.rectangle([x1, text_y, x1 + 120, text_y + 17], fill=color)
        draw.text((x1 + 3, text_y + 1), label, fill=(255, 255, 255), font=small_font)

    return canvas

# Collect image-level errors
rows = []
image_paths = sorted([p for p in IMG_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTS])

for img_path in image_paths:
    gt_path = GT_DIR / f"{img_path.stem}.txt"
    pred_path = PRED_DIR / f"{img_path.stem}.txt"

    gt_boxes = read_yolo_labels(gt_path, is_pred=False)
    pred_boxes = read_yolo_labels(pred_path, is_pred=True)

    row = evaluate_image(img_path, gt_boxes, pred_boxes)
    rows.append(row)

# Save CSV
with open(OUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(sorted(rows, key=lambda r: r["score"], reverse=True))

# Select examples: prioritize wrong classes/hallucinations, then many misses.
sorted_rows = sorted(rows, key=lambda r: (r["wrong_class"], r["hallucinated_non_overlap_classes"], r["false_negatives"], r["false_positives"]), reverse=True)

selected = []
seen = set()

for r in sorted_rows:
    if r["image"] not in seen and len(selected) < 6:
        selected.append(r)
        seen.add(r["image"])

# Build contact sheet
row_canvases = []

for r in selected:
    img_path = IMG_DIR / r["image"]
    gt_path = GT_DIR / f"{img_path.stem}.txt"
    pred_path = PRED_DIR / f"{img_path.stem}.txt"

    gt_boxes = read_yolo_labels(gt_path, is_pred=False)
    pred_boxes = read_yolo_labels(pred_path, is_pred=True)

    gt_panel = draw_boxes(
        img_path,
        gt_boxes,
        f"{r['image']} | Ground truth ({r['n_gt']} objects)",
        mode="gt",
    )

    pred_panel = draw_boxes(
        img_path,
        pred_boxes,
        f"Model predictions | FN={r['false_negatives']}, FP={r['false_positives']}, wrong={r['wrong_class']}, hallucinated={r['hallucinated_non_overlap_classes']}",
        mode="pred",
    )

    gap = 20
    row_w = gt_panel.width + pred_panel.width + gap
    row_h = max(gt_panel.height, pred_panel.height)

    row_canvas = Image.new("RGB", (row_w, row_h), "white")
    row_canvas.paste(gt_panel, (0, 0))
    row_canvas.paste(pred_panel, (gt_panel.width + gap, 0))

    row_canvases.append(row_canvas)

gap_y = 25
total_w = max(r.width for r in row_canvases)
total_h = sum(r.height for r in row_canvases) + gap_y * (len(row_canvases) - 1)

sheet = Image.new("RGB", (total_w, total_h), "white")

y = 0
for row in row_canvases:
    sheet.paste(row, (0, y))
    y += row.height + gap_y

sheet.save(OUT_IMAGE, quality=95)

print("Saved qualitative panel:", OUT_IMAGE)
print("Saved error CSV:", OUT_CSV)
