#!/usr/bin/env python3
"""
score_single_complete.py

Compute TRUE single-view scores v_single for each (instance_id, base_class, view_id)
by matching YOLO predictions (from pred_cache JSON) against GT boxes from a manifest.

- Reads manifest CSV with columns:
  instance_id, base_class, view_id, image_path, gt_x1, gt_y1, gt_x2, gt_y2
- Reads per-image prediction JSON from:
  pred_dir/<stem(image_path)>.json

Expected pred_cache JSON format (your case):
[
  {"xyxy":[x1,y1,x2,y2], "conf":..., "cls": int},
  ...
]

Outputs CSV with columns:
instance_id, base_class, view_id, image_path, v_single

Scoring:
For each GT box (rows for that image/instance/class/view), take best IoU
among predictions of the correct class-id; then v_single is the maximum IoU
across GT boxes for that (instance_id, base_class, view_id, image_path) group.

This script is "fail-loud": it prints coverage stats and counts missing json.
"""

import os
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from ultralytics import YOLO


def stem(p: str) -> str:
    return os.path.splitext(os.path.basename(p))[0]


def iou_xyxy(a, b) -> float:
    # a,b: (x1,y1,x2,y2)
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def load_preds_any(pred_json_path: str):
    """
    Returns a list of predictions in one of these normalized forms:
      - dict preds: {"xyxy":[...], "conf": float, "cls": int}
      - fallback list preds: [x1,y1,x2,y2] (cls unknown)
    """
    with open(pred_json_path, "r") as f:
        d = json.load(f)

    # Most common: list of dicts with xyxy/conf/cls
    if isinstance(d, list):
        if len(d) == 0:
            return []
        if isinstance(d[0], dict) and "xyxy" in d[0]:
            return d
        # list of boxes
        if isinstance(d[0], (list, tuple)) and len(d[0]) == 4:
            return d

    # Sometimes wrapped in a dict
    if isinstance(d, dict):
        # common wrappers
        for key in ("preds", "detections", "boxes", "results", "predictions"):
            if key in d:
                v = d[key]
                if isinstance(v, list):
                    if len(v) == 0:
                        return []
                    if isinstance(v[0], dict) and "xyxy" in v[0]:
                        return v
                    if isinstance(v[0], (list, tuple)) and len(v[0]) == 4:
                        return v

        # ultralytics export variants: {"pred": [...]} etc.
        if "pred" in d and isinstance(d["pred"], list):
            return d["pred"]

    raise ValueError(f"Unrecognized pred json format: {pred_json_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="CSV manifest with GT rows")
    ap.add_argument("--pred_dir", required=True, help="Directory with cached YOLO pred JSONs")
    ap.add_argument("--weights", required=True, help="YOLO weights (.pt) to read class-name mapping")
    ap.add_argument("--out_csv", required=True, help="Output CSV path")
    ap.add_argument("--score_thr", type=float, default=0.0, help="Optional confidence threshold for preds")
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)

    needed = ["instance_id", "base_class", "view_id", "image_path", "gt_x1", "gt_y1", "gt_x2", "gt_y2"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Manifest missing columns {missing}. Have: {df.columns.tolist()}")

    # Load YOLO mapping: base_class (string) -> class id (int)
    yolo = YOLO(args.weights)
    names = yolo.model.names
    if not isinstance(names, dict):
        names = {i: n for i, n in enumerate(names)}
    name_to_id = {v: k for k, v in names.items()}

    # Quick sanity: manifest classes should be known
    manifest_classes = sorted(df["base_class"].dropna().unique().tolist())
    unknown = [c for c in manifest_classes if c not in name_to_id]
    if unknown:
        raise ValueError(
            f"Some manifest base_class values are not in YOLO names: {unknown}\n"
            f"YOLO names: {names}"
        )

    # Group by single-view identity. Keep image_path in the group key to avoid mixing if any duplicates exist.
    group_cols = ["instance_id", "base_class", "view_id", "image_path"]
    groups = df.groupby(group_cols, sort=False)

    cache = {}
    missing_json = 0
    read_errors = 0
    no_preds_after_thr = 0
    total_groups = len(groups)

    rows_out = []

    for (inst, base_class, view_id, img_path), g in tqdm(groups, total=total_groups, desc="scoring singles"):
        st = stem(str(img_path))
        jpath = os.path.join(args.pred_dir, f"{st}.json")

        if st not in cache:
            if not os.path.exists(jpath):
                cache[st] = None
            else:
                try:
                    cache[st] = load_preds_any(jpath)
                except Exception:
                    cache[st] = None
                    read_errors += 1

        preds = cache[st]
        if preds is None:
            missing_json += 1
            best = 0.0
        else:
            # Optional score threshold
            if args.score_thr > 0:
                if len(preds) and isinstance(preds[0], dict):
                    preds_f = [p for p in preds if float(p.get("conf", 0.0)) >= args.score_thr]
                else:
                    preds_f = preds  # can't threshold if no conf
            else:
                preds_f = preds

            if len(preds_f) == 0:
                no_preds_after_thr += 1
                best = 0.0
            else:
                gt_id = int(name_to_id[base_class])
                best = 0.0

                # evaluate each GT row; take best IoU among matching-class preds
                for _, r in g.iterrows():
                    gt = (r["gt_x1"], r["gt_y1"], r["gt_x2"], r["gt_y2"])

                    # find best pred IoU for this GT
                    best_gt = 0.0
                    for p in preds_f:
                        if isinstance(p, dict):
                            # enforce class match
                            if int(p.get("cls", -999)) != gt_id:
                                continue
                            box = p["xyxy"]
                        else:
                            # fallback: no class available => cannot class-filter reliably
                            # We still allow it, but note: this can inflate scores if used.
                            box = p

                        val = iou_xyxy(gt, box)
                        if val > best_gt:
                            best_gt = val

                    if best_gt > best:
                        best = best_gt

        rows_out.append(
            {
                "instance_id": inst,
                "base_class": base_class,
                "view_id": view_id,
                "image_path": img_path,
                "v_single": float(best),
            }
        )

    out = pd.DataFrame(rows_out)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    # Summary stats
    v = out["v_single"].to_numpy()
    print("\n=== Single scoring summary ===")
    print("out_csv:", args.out_csv)
    print("rows:", len(out))
    print("unique instances:", out["instance_id"].nunique())
    print("unique views:", out["view_id"].nunique())
    print("missing_json groups:", missing_json, f"({missing_json/total_groups:.2%})")
    print("read_errors:", read_errors)
    print("no_preds_after_thr groups:", no_preds_after_thr, f"(thr={args.score_thr})")
    print("v_single min/mean/max:", float(np.min(v)), float(np.mean(v)), float(np.max(v)))
    print("fraction v_single==0:", float(np.mean(v == 0.0)))

    # Extra: show per-class mean
    per_class = out.groupby("base_class")["v_single"].mean().sort_values(ascending=False)
    print("\nper-class mean v_single:")
    print(per_class.to_string())


if __name__ == "__main__":
    main()
