import os
import json
import argparse
import pandas as pd
from itertools import combinations

def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2-ax1) * max(0.0, ay2-ay1)
    area_b = max(0.0, bx2-bx1) * max(0.0, by2-by1)
    union = area_a + area_b - inter + 1e-9
    return inter / union

def load_preds(pred_dir, image_path):
    stem = os.path.splitext(os.path.basename(image_path))[0]
    p = os.path.join(pred_dir, f"{stem}.json")
    if not os.path.exists(p):
        return []
    with open(p, "r") as f:
        return json.load(f)

def best_iou(preds, gt_xyxy, gt_cls=None):
    best = 0.0
    for d in preds:
        if gt_cls is not None and int(d.get("cls", -1)) != int(gt_cls):
            continue
        best = max(best, iou_xyxy(d["xyxy"], gt_xyxy))
    return float(best)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--out_single", required=True)
    ap.add_argument("--out_pairs", required=True)
    ap.add_argument("--out_best_per_class", required=True)
    ap.add_argument("--filter_by_class", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    print("[debug] manifest columns:", df.columns.tolist())
    print("[debug] manifest rows:", len(df))

    # image-level rows only
    df = df.drop_duplicates(subset=["image_path"]).copy()
    print("[debug] unique images:", len(df))

    # robust instance key from filename (before '-el')
    df["instance_key"] = df["image_path"].apply(lambda p: os.path.basename(p).split("-el")[0])

    # build single-view table
    single_rows = []
    missing_pred = 0

    for row in df.itertuples(index=False):
        gt = [row.gt_x1, row.gt_y1, row.gt_x2, row.gt_y2]
        preds = load_preds(args.pred_dir, row.image_path)
        if preds == []:
            missing_pred += 1

        v = best_iou(preds, gt, row.gt_category_id if args.filter_by_class else None)

        single_rows.append({
            "instance_id": row.instance_key,
            "base_class": row.base_class,
            "view_id": row.view_id,
            "image_path": row.image_path,
            "v_single": v
        })

    single_df = pd.DataFrame(single_rows)
    print("[debug] missing pred files:", missing_pred, "/", len(single_df))
    print("[debug] single_df columns:", single_df.columns.tolist())
    print("[debug] unique instances:", single_df["instance_id"].nunique())
    print("[debug] views per instance (describe):")
    print(single_df.groupby("instance_id")["view_id"].count().describe())

    single_df.to_csv(args.out_single, index=False)

    # compute pairs per instance
    pair_rows = []
    for inst, g in single_df.groupby("instance_id"):
        views = g[["view_id", "v_single"]].dropna().values.tolist()
        if len(views) < 2:
            continue
        cls_name = g["base_class"].iloc[0]
        for (va, sa), (vb, sb) in combinations(views, 2):
            pair_rows.append({
                "instance_id": inst,
                "base_class": cls_name,
                "view_a": va,
                "view_b": vb,
                "v_a": float(sa),
                "v_b": float(sb),
                "v_ab": float(max(sa, sb))
            })

    pair_df = pd.DataFrame(pair_rows)
    print("[debug] pair_df rows:", len(pair_df))
    print("[debug] pair_df columns:", pair_df.columns.tolist())

    if len(pair_df) == 0:
        print("\n[ERROR] No pairs were formed.")
        print("Most likely: each instance has <2 views after parsing, or view_id is missing.")
        print("Check the views-per-instance describe above.")
        return

    pair_df.to_csv(args.out_pairs, index=False)

    # best pair per class + exact 2-player shapley
    best_rows = []
    for cls, g in pair_df.groupby("base_class"):
        best = g.loc[g["v_ab"].idxmax()]
        phi_a = 0.5*best.v_a + 0.5*(best.v_ab - best.v_b)
        phi_b = 0.5*best.v_b + 0.5*(best.v_ab - best.v_a)
        best_rows.append({
            "base_class": cls,
            "best_pair": f"({best.view_a}, {best.view_b})",
            "v_ab": float(best.v_ab),
            "phi_a": float(phi_a),
            "phi_b": float(phi_b),
            "dominant_view": best.view_a if phi_a >= phi_b else best.view_b
        })

    best_df = pd.DataFrame(best_rows).sort_values("v_ab", ascending=False)
    best_df.to_csv(args.out_best_per_class, index=False)
    print("\n[debug] best per class:")
    print(best_df)

if __name__ == "__main__":
    main()
