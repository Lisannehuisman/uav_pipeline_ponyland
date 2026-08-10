#!/usr/bin/env python3
import os, json
import pandas as pd
import numpy as np
from ultralytics import YOLO

# EDIT THESE if needed (but they should work if you export OUT/W in your shell)
MAN = "manifests/val_M4_manifest.csv"
PRED_DIR = os.environ.get("PRED_DIR", "")
WEIGHTS = os.environ.get("W", "")

def stem(p: str) -> str:
    return os.path.splitext(os.path.basename(p))[0]

def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    bb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = aa + bb - inter
    return 0.0 if denom <= 0 else inter / denom

def main():
    if not WEIGHTS or not os.path.exists(WEIGHTS):
        raise SystemExit(f"WEIGHTS not set or not found. Set env var W to your best.pt. Got: {WEIGHTS}")
    if not PRED_DIR or not os.path.isdir(PRED_DIR):
        raise SystemExit(f"PRED_DIR not set or not found. Set env var PRED_DIR to your pred_cache dir. Got: {PRED_DIR}")

    y = YOLO(WEIGHTS)
    name2id = {str(v): int(k) for k, v in y.names.items()}

    m = pd.read_csv(MAN)
    n = min(800, len(m))
    m = m.sample(n=n, random_state=0).reset_index(drop=True)

    best_any = []
    best_correct = []
    has_correct = 0
    bad_pred_cls = 0
    missing_json = 0

    for _, r in m.iterrows():
        img = r["image_path"]
        jpath = os.path.join(PRED_DIR, f"{stem(img)}.json")
        if not os.path.exists(jpath):
            missing_json += 1
            best_any.append(0.0)
            best_correct.append(0.0)
            continue

        d = json.load(open(jpath, "r"))
        preds = d.get("preds", [])

        gt = [float(r["gt_x1"]), float(r["gt_y1"]), float(r["gt_x2"]), float(r["gt_y2"])]
        gt_cls = name2id[str(r["base_class"])]

        bi = 0.0
        for p in preds:
            bi = max(bi, iou_xyxy(p["xyxy"], gt))
        best_any.append(bi)

        bc = 0.0
        found = False
        for p in preds:
            try:
                pc = int(p["cls"])
            except Exception:
                bad_pred_cls += 1
                continue
            if pc == gt_cls:
                found = True
                bc = max(bc, iou_xyxy(p["xyxy"], gt))
        best_correct.append(bc)
        if found:
            has_correct += 1

    best_any = np.array(best_any)
    best_correct = np.array(best_correct)

    print("MAN:", MAN)
    print("PRED_DIR:", PRED_DIR)
    print("WEIGHTS:", WEIGHTS)
    print("Samples:", len(m))
    print("missing_json in sample:", missing_json)
    print("GT class present among preds:", has_correct, f"({has_correct/len(m):.3f})")
    print("bad pred cls parse count:", bad_pred_cls)

    print("\nBest IoU ANY:")
    print(" mean:", float(best_any.mean()),
          " median:", float(np.median(best_any)),
          " frac>0:", float((best_any > 0).mean()))

    print("\nBest IoU CORRECT CLASS:")
    print(" mean:", float(best_correct.mean()),
          " median:", float(np.median(best_correct)),
          " frac>0:", float((best_correct > 0).mean()))

    if (best_any > 0).any():
        print("\nMean IoU ANY given >0:", float(best_any[best_any > 0].mean()))
    if (best_correct > 0).any():
        print("Mean IoU CORRECT given >0:", float(best_correct[best_correct > 0].mean()))

if __name__ == "__main__":
    main()
