#!/usr/bin/env python3
import os, json, argparse
import pandas as pd
from ultralytics import YOLO

def stem(p: str) -> str:
    return os.path.splitext(os.path.basename(p))[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--device", default="0")
    ap.add_argument("--limit", type=int, default=0, help="0=no limit; else first N unique images")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    m = pd.read_csv(args.manifest)
    assert "image_path" in m.columns, "manifest must have column image_path"

    imgs = m["image_path"].dropna().astype(str).unique().tolist()
    if args.limit and args.limit > 0:
        imgs = imgs[:args.limit]

    model = YOLO(args.weights)

    missing = 0
    for idx, img_path in enumerate(imgs, start=1):
        if not os.path.exists(img_path):
            missing += 1
            continue

        out_path = os.path.join(args.outdir, f"{stem(img_path)}.json")

        res_list = model.predict(
            source=img_path,
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device,
            verbose=False,
            task="detect",
        )

        r = res_list[0]
        preds = []
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clss  = r.boxes.cls.cpu().numpy().astype(int)
            for bb, c, k in zip(xyxy, confs, clss):
                preds.append({
                    "xyxy": [float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])],
                    "conf": float(c),
                    "cls": int(k),
                })

        payload = {"image_path": img_path, "imgsz": args.imgsz, "conf": args.conf, "preds": preds}
        with open(out_path, "w") as f:
            json.dump(payload, f)

        if idx % 500 == 0:
            print(f"[{idx}/{len(imgs)}] wrote {out_path}")

    print("\n=== cache preds summary ===")
    print("unique images:", len(imgs))
    print("missing images:", missing)
    print("outdir:", args.outdir)

if __name__ == "__main__":
    main()
