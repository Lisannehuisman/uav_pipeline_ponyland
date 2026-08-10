#!/usr/bin/env python3
from ultralytics import YOLO
from pathlib import Path
import argparse, time, json, os, socket

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset yaml")
    ap.add_argument("--regime", required=True, help="e.g. M1, M2a, ...")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--project", default="/vol/tensusers6/lisannehuisman/projects/yolov8n/runs_yolov8n")
    args = ap.parse_args()

    project = Path(args.project)
    project.mkdir(parents=True, exist_ok=True)

    run_name = f"S0_{args.regime}_yolov8n"
    model = YOLO("yolov8n.pt")

    t0 = time.time()
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(project),
        name=run_name,
        workers=4,
        verbose=True,
    )
    wall = time.time() - t0

    meta = {
        "host": socket.gethostname(),
        "regime": args.regime,
        "data_yaml": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "wall_seconds": wall,
        "wall_hours": wall/3600.0,
        "project": str(project),
        "run_name": run_name,
    }

    # store alongside run output
    meta_path = project / run_name / "train_walltime.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print("Saved wall time:", meta_path)

if __name__ == "__main__":
    main()

