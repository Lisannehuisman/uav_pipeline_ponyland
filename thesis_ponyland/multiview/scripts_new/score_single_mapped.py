#!/usr/bin/env python3
import os, json, argparse
import numpy as np
import pandas as pd
from ultralytics import YOLO

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
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--score_thr", type=float, default=0.0)
    ap.add_argument("--ignore_class", action="store_true")
    args = ap.parse_args()

    m = pd.read_csv(args.manifest)

    required = ("image_path","gt_x1","gt_y1","gt_x2","gt_y2","base_class")
    if not all(c in m.columns for c in required):
        raise ValueError(f"Manifest missing required columns. Found: {m.columns.tolist()}")

    yolo = YOLO(args.weights)
    names = yolo.names

    if not isinstance(names, dict):
        raise ValueError("Expected yolo.names to be a dict")

    name2id = {str(v): int(k) for k, v in names.items()}

    if not args.ignore_class:
        missing = sorted(set(m["base_class"].astype(str).unique()) - set(name2id.keys()))
        if missing:
            raise ValueError("base_class not found in YOLO names: " + ", ".join(missing))

    rows = []
    missing_json = 0
    read_errors = 0
    no_preds_after_thr = 0

    grouped = m.groupby("image_path", sort=False)
    total_groups = len(grouped)

    for img_path, sub in grouped:
        st = stem(img_path)
        jpath = os.path.join(args.pred_dir, f"{st}.json")

        if not os.path.exists(jpath):
            missing_json += 1
            continue

        try:
            with open(jpath, "r") as f:
                d = json.load(f)
            preds = d.get("preds", [])
        except Exception:
            read_errors += 1
            continue

        preds_thr = [p for p in preds if p.get("conf", 0.0) >= args.score_thr]
        if len(preds_thr) == 0:
            no_preds_after_thr += 1

        for _, r in sub.iterrows():
            gt_xyxy = [float(r["gt_x1"]), float(r["gt_y1"]),
                       float(r["gt_x2"]), float(r["gt_y2"])]

            gt_cls = None
            if not args.ignore_class:
                gt_cls = name2id[str(r["base_class"])]

            best = 0.0
            for p in preds_thr:
                if (gt_cls is not None) and int(p["cls"]) != gt_cls:
                    continue
                iou = iou_xyxy(p["xyxy"], gt_xyxy)
                if iou > best:
                    best = iou

            out = dict(r)
            out["v_single"] = float(best)
            rows.append(out)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out_csv, index=False)

    v = out_df["v_single"].to_numpy()

    print("\n=== Single scoring summary (MAPPED) ===")
    print("rows:", len(out_df))
    print("missing_json groups:", missing_json)
    print("read_errors:", read_errors)
    print("no_preds_after_thr groups:", no_preds_after_thr)
    print("mean IoU:", float(v.mean()))
    print("fraction zero:", float((v==0).mean()))

if __name__ == "__main__":
    main()
