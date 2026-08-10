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

    required = ("image_path","gt_x1","gt_y1","gt_x2","gt_y2","gt_category_id")
    if not all(c in m.columns for c in required):
        raise ValueError(f"Manifest missing required columns. Found: {m.columns.tolist()}")

    yolo = YOLO(args.weights)
    names = yolo.names if hasattr(yolo, "names") else None

    rows = []
    missing_json = 0
    read_errors = 0
    no_preds_after_thr = 0

    g = m.groupby("image_path", sort=False)
    total_groups = len(g)

    for img_path, sub in g:
        st = stem(img_path)
        jpath = os.path.join(args.pred_dir, f"{st}.json")
        if not os.path.exists(jpath):
            missing_json += 1
            continue
        try:
            d = json.load(open(jpath, "r"))
            preds = d.get("preds", [])
        except Exception:
            read_errors += 1
            continue

        preds_thr = [p for p in preds if p.get("conf", 0.0) >= args.score_thr]
        if len(preds_thr) == 0:
            no_preds_after_thr += 1

        for _, r in sub.iterrows():
            gt_xyxy = [float(r["gt_x1"]), float(r["gt_y1"]), float(r["gt_x2"]), float(r["gt_y2"])]
            gt_cls = int(r["gt_category_id"])

            best = 0.0
            for p in preds_thr:
                if (not args.ignore_class) and int(p["cls"]) != gt_cls:
                    continue
                iou = iou_xyxy(p["xyxy"], gt_xyxy)
                if iou > best:
                    best = iou

            out = dict(r)
            out["v_single"] = float(best)
            out["gt_cls_name"] = names.get(gt_cls, str(gt_cls)) if isinstance(names, dict) else str(gt_cls)
            rows.append(out)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out_csv, index=False)

    v = out_df["v_single"].to_numpy() if len(out_df) else np.array([0.0])
    print("\n=== Single scoring summary (NEW) ===")
    print("out_csv:", args.out_csv)
    print("rows:", len(out_df))
    print("missing_json groups:", missing_json, f"({missing_json/total_groups:.2%})")
    print("read_errors:", read_errors)
    print("no_preds_after_thr groups:", no_preds_after_thr, f"(thr={args.score_thr})")
    print("v_single min/mean/max:", float(v.min()), float(v.mean()), float(v.max()))
    print("fraction v_single==0:", float((v==0).mean()))
    if "base_class" in out_df.columns:
        print("\nper-class mean v_single:")
        print(out_df.groupby("base_class")["v_single"].mean().sort_values(ascending=False))

if __name__ == "__main__":
    main()
