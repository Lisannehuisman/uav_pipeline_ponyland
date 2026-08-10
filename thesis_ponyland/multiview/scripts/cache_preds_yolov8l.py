import os
import json
import argparse
import pandas as pd
from ultralytics import YOLO
from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.manifest)
    model = YOLO(args.weights)

    for _, row in tqdm(df.iterrows(), total=len(df)):
        img = row["image_path"]

        # unique key per image
        stem = os.path.splitext(os.path.basename(img))[0]
        out_name = f"{stem}.json"
        out_path = os.path.join(args.outdir, out_name)

        if os.path.exists(out_path):
            continue

        res = model.predict(
            source=img,
            imgsz=args.imgsz,
            conf=0.001,
            iou=0.7,
            device=0,
            verbose=False
        )[0]

        preds = []
        if res.boxes is not None:
            for b in res.boxes:
                preds.append({
                    "xyxy": b.xyxy[0].tolist(),
                    "conf": float(b.conf),
                    "cls": int(b.cls)
                })

        with open(out_path, "w") as f:
            json.dump(preds, f)


if __name__ == "__main__":
    main()
