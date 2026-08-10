#!/usr/bin/env python3
import argparse, json, os, re
import pandas as pd

def coco_bbox_to_xyxy(b):
    # COCO bbox is [x, y, w, h]
    x, y, w, h = b
    return x, y, x + w, y + h

def parse_ids_from_filename(fname: str):
    """
    Expected filename example:
      S0-SM_barrel_1-elhigh-radmid-az135.png

    Returns:
      instance_id = S0-SM_barrel_1
      view_id     = elhigh-radmid-az135
    """
    base = os.path.splitext(os.path.basename(fname))[0]

    if "-el" not in base:
        return base, None

    instance_id = base.split("-el", 1)[0]
    view_id = "el" + base.split("-el", 1)[1]
    return instance_id, view_id



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco_json", required=True, help="Path to COCO annotations json")
    ap.add_argument("--images_root", required=True, help="Root directory where COCO 'file_name' lives")
    ap.add_argument("--out", required=True, help="Output parquet/csv path")
    ap.add_argument("--out_format", default="parquet", choices=["parquet","csv"])
    args = ap.parse_args()

    with open(args.coco_json, "r") as f:
        coco = json.load(f)

    # category id -> name
    cat_map = {c["id"]: c["name"] for c in coco["categories"]}

    # image_id -> file_name
    img_map = {im["id"]: im["file_name"] for im in coco["images"]}

    rows = []
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        file_name = img_map[img_id]
        image_path = os.path.join(args.images_root, file_name)

        instance_id, view_id = parse_ids_from_filename(file_name)
        x1, y1, x2, y2 = coco_bbox_to_xyxy(ann["bbox"])
        base_class = cat_map.get(ann["category_id"], str(ann["category_id"]))

        rows.append({
            "image_id": img_id,
            "file_name": file_name,
            "image_path": image_path,
            "instance_id": instance_id,
            "view_id": view_id,
            "base_class": base_class,
            "gt_x1": x1, "gt_y1": y1, "gt_x2": x2, "gt_y2": y2,
            "gt_category_id": ann["category_id"],
        })

    df = pd.DataFrame(rows)

    # Basic sanity
    missing = df["view_id"].isna().mean()
    print(f"[manifest] rows={len(df)} unique_instances={df['instance_id'].nunique()} missing_view_id_frac={missing:.3f}")

    if args.out_format == "parquet":
        df.to_parquet(args.out, index=False)
    else:
        df.to_csv(args.out, index=False)

    print(f"[manifest] wrote {args.out}")

if __name__ == "__main__":
    main()

