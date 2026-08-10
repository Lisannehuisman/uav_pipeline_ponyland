from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

os.environ.setdefault(
    "YOLO_CONFIG_DIR",
    str((Path(__file__).resolve().parents[1] / "Ultralytics").resolve()),
)
from ultralytics import YOLO

from comparison_config import DEFAULT_OUTPUT_DIR, DETECTOR_ORDER, MODEL_RUNS, REGIME_DATA_YAMLS, REGIME_ORDER

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except ImportError as exc:
    raise SystemExit(
        "pycocotools is required for standardized_test_eval.py. "
        "Install it with `pip install pycocotools` in your active environment."
    ) from exc


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SUMMARY_FIELDS = [
    "detector",
    "regime",
    "precision",
    "recall",
    "f1",
    "map50",
    "map50_95",
    "ap75",
    "matched_mean_iou",
]


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else float("nan")


def resolve_frcnn_run_dir(run_dir: Path) -> Path:
    if (run_dir / "inference" / "coco_instances_results.json").exists():
        return run_dir
    nested_candidates = list(run_dir.glob("*/inference/coco_instances_results.json"))
    if nested_candidates:
        return nested_candidates[0].parent.parent
    return run_dir


def cached_prediction_candidates(pred_dir: Path, detector: str, regime: str, split: str) -> list[Path]:
    base = detector.replace(" ", "_")
    if detector.startswith("YOLO"):
        return [
            pred_dir / f"{base}_{regime}_{split}_predictions.json",
            pred_dir / f"{base}_{regime}_predictions.json",
        ]
    return [
        pred_dir / f"{base}_{regime}_{split}_predictions.json",
        pred_dir / f"{base}_{regime}_predictions.json",
    ]


def iter_chunks(values: list[Path], chunk_size: int) -> Iterable[list[Path]]:
    for start in range(0, len(values), chunk_size):
        yield values[start : start + chunk_size]


def resolve_dataset_root(data_yaml: Path, data_dict: dict) -> Path:
    configured_root = data_dict.get("path")
    if configured_root:
        root = Path(configured_root)
        if not root.is_absolute():
            root = (data_yaml.parent / root).resolve()
        return root
    return data_yaml.parent.resolve()


def resolve_split_images(data_yaml: Path, split: str = "test") -> tuple[dict, list[Path]]:
    with data_yaml.open("r", encoding="utf-8") as handle:
        data_dict = yaml.safe_load(handle)

    split_value = data_dict.get(split)
    if split_value is None:
        raise ValueError(f"Split '{split}' was not found in {data_yaml}.")

    root = resolve_dataset_root(data_yaml, data_dict)
    candidates = [Path(split_value)] if isinstance(split_value, str) else [Path(item) for item in split_value]
    images: list[Path] = []

    for candidate in candidates:
        resolved = candidate if candidate.is_absolute() else (root / candidate).resolve()
        if resolved.is_dir():
            for path in resolved.rglob("*"):
                if path.suffix.lower() in IMAGE_EXTENSIONS:
                    images.append(path)
        elif resolved.is_file() and resolved.suffix.lower() == ".txt":
            with resolved.open("r", encoding="utf-8") as handle:
                for line in handle:
                    raw = line.strip()
                    if not raw:
                        continue
                    listed = Path(raw)
                    if not listed.is_absolute():
                        listed = (resolved.parent / listed).resolve()
                    if listed.suffix.lower() in IMAGE_EXTENSIONS:
                        images.append(listed)
        elif resolved.is_file() and resolved.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(resolved)
        else:
            raise FileNotFoundError(f"Could not resolve image path '{candidate}' from split '{split}'.")

    return data_dict, sorted(images)


def label_path_from_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    for idx, part in enumerate(parts):
        if part == "images":
            parts[idx] = "labels"
            return Path(*parts).with_suffix(".txt")
    raise ValueError(f"Could not derive YOLO label path from {image_path}.")


def image_size(image_path: Path) -> tuple[int, int]:
    image = plt.imread(str(image_path))
    if image.ndim == 2:
        height, width = image.shape
    else:
        height, width = image.shape[:2]
    return width, height


def load_yolo_annotations(label_path: Path, width: int, height: int) -> list[dict[str, float | int]]:
    if not label_path.exists():
        return []
    rows = []
    with label_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            class_id, xc, yc, w, h = parts
            class_id = int(float(class_id))
            xc = float(xc) * width
            yc = float(yc) * height
            w = float(w) * width
            h = float(h) * height
            x = xc - w / 2
            y = yc - h / 2
            rows.append(
                {
                    "category_id": class_id + 1,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                }
            )
    return rows


def build_coco_gt(data_yaml: Path, out_json: Path, split: str) -> tuple[list[Path], dict[str, int]]:
    data_dict, image_paths = resolve_split_images(data_yaml, split=split)
    names = data_dict.get("names", {})
    if isinstance(names, dict):
        categories = [{"id": int(class_id) + 1, "name": str(class_name)} for class_id, class_name in names.items()]
    else:
        categories = [{"id": idx + 1, "name": str(class_name)} for idx, class_name in enumerate(names)]

    coco_images = []
    coco_annotations = []
    image_id_map: dict[str, int] = {}
    annotation_id = 1

    for image_id, image_path in enumerate(image_paths, start=1):
        width, height = image_size(image_path)
        image_id_map[str(image_path)] = image_id
        coco_images.append({"id": image_id, "file_name": image_path.name, "width": width, "height": height})

        label_path = label_path_from_image(image_path)
        for ann in load_yolo_annotations(label_path, width, height):
            coco_annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    **ann,
                }
            )
            annotation_id += 1

    coco_dict = {
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": categories,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(coco_dict), encoding="utf-8")
    return image_paths, image_id_map


def load_or_build_coco_gt(data_yaml: Path, out_json: Path, split: str) -> tuple[list[Path], dict[str, int]]:
    _, image_paths = resolve_split_images(data_yaml, split=split)
    image_id_map = {str(image_path): image_id for image_id, image_path in enumerate(image_paths, start=1)}
    if out_json.exists():
        return image_paths, image_id_map
    return build_coco_gt(data_yaml, out_json, split=split)


def resolve_official_coco_gt(data_yaml: Path, split: str) -> Path | None:
    if split != "val":
        return None

    with data_yaml.open("r", encoding="utf-8") as handle:
        data_dict = yaml.safe_load(handle)

    dataset_root = resolve_dataset_root(data_yaml, data_dict)
    regime_name = data_yaml.stem
    candidates = [
        dataset_root / "coco_annotations" / f"coco_instances_{split}_{regime_name}.json",
        dataset_root / "annotations" / f"instances_{split}_{regime_name}.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_paths_from_official_coco_gt(gt_json: Path, data_yaml: Path, split: str) -> tuple[list[Path], dict[str, int]]:
    gt = json.loads(gt_json.read_text(encoding="utf-8"))
    _, available_images = resolve_split_images(data_yaml, split=split)
    images_by_name = {image_path.name: image_path for image_path in available_images}

    ordered_paths: list[Path] = []
    image_id_map: dict[str, int] = {}
    for image_info in gt["images"]:
        file_name = str(image_info["file_name"])
        if file_name not in images_by_name:
            raise FileNotFoundError(f"Could not match COCO image '{file_name}' to the '{split}' split for {data_yaml}.")
        image_path = images_by_name[file_name]
        ordered_paths.append(image_path)
        image_id_map[str(image_path)] = int(image_info["id"])

    return ordered_paths, image_id_map


def predict_yolo_to_coco_json(
    weights_path: Path,
    image_paths: list[Path],
    image_id_map: dict[str, int],
    out_json: Path,
    imgsz: int = 640,
    conf: float = 0.001,
    batch: int = 16,
    device: str | None = None,
) -> None:
    model = YOLO(str(weights_path))
    results_json = []

    for batch_paths in iter_chunks(image_paths, batch):
        results = model.predict(
            source=[str(path) for path in batch_paths],
            imgsz=imgsz,
            conf=conf,
            device=device,
            verbose=False,
        )
        for image_path, result in zip(batch_paths, results):
            if result.boxes is None or len(result.boxes) == 0:
                continue
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)

            for box, score, class_id in zip(boxes, scores, classes):
                x1, y1, x2, y2 = box.tolist()
                results_json.append(
                    {
                        "image_id": image_id_map[str(image_path)],
                        "category_id": int(class_id) + 1,
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": float(score),
                    }
                )

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results_json), encoding="utf-8")


def coco_precision_recall_f1(ev: COCOeval) -> tuple[float, float, float]:
    iou_thresholds = ev.params.iouThrs
    iou_idx = int(np.argmin(np.abs(iou_thresholds - 0.5)))
    precision = ev.eval["precision"][iou_idx, :, :, 0, -1]
    recall = ev.eval["recall"][iou_idx, :, 0, -1]

    precision_values = precision[precision > -1]
    recall_values = recall[recall > -1]

    p50 = float(np.mean(precision_values)) if precision_values.size else float("nan")
    r50 = float(np.mean(recall_values)) if recall_values.size else float("nan")
    f1_50 = safe_divide(2 * p50 * r50, p50 + r50)
    return p50, r50, f1_50


def load_coco_json(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def coco_bbox_to_xyxy(bbox: list[float]) -> tuple[float, float, float, float]:
    x, y, w, h = bbox
    return x, y, x + w, y + h


def iou_xyxy(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def matched_mean_iou(gt_json: Path, pred_json: Path, score_threshold: float = 0.001) -> float:
    gt = json.loads(gt_json.read_text(encoding="utf-8"))
    preds = load_coco_json(pred_json)

    gt_by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in gt["annotations"]:
        gt_by_image[int(ann["image_id"])].append(ann)

    pred_by_image: dict[int, list[dict]] = defaultdict(list)
    for pred in preds:
        if float(pred["score"]) >= score_threshold:
            pred_by_image[int(pred["image_id"])].append(pred)

    matched_ious: list[float] = []
    for image_id, image_preds in pred_by_image.items():
        image_gts = gt_by_image.get(image_id, [])
        image_preds = sorted(image_preds, key=lambda item: float(item["score"]), reverse=True)
        matched_gt_ids: set[int] = set()

        for pred in image_preds:
            best_iou = 0.0
            best_gt = None
            pred_box = coco_bbox_to_xyxy(pred["bbox"])
            pred_class = int(pred["category_id"])
            for gt_ann in image_gts:
                if int(gt_ann["id"]) in matched_gt_ids or int(gt_ann["category_id"]) != pred_class:
                    continue
                gt_box = coco_bbox_to_xyxy(gt_ann["bbox"])
                current_iou = iou_xyxy(pred_box, gt_box)
                if current_iou >= 0.5 and current_iou > best_iou:
                    best_iou = current_iou
                    best_gt = gt_ann
            if best_gt is not None:
                matched_gt_ids.add(int(best_gt["id"]))
                matched_ious.append(best_iou)

    return float(np.mean(matched_ious)) if matched_ious else float("nan")


def evaluate_coco(gt_json: Path, pred_json: Path) -> dict[str, float]:
    coco_gt = COCO(str(gt_json))
    coco_dt = coco_gt.loadRes(str(pred_json))
    ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    stats = ev.stats
    p50, r50, f1_50 = coco_precision_recall_f1(ev)
    return {
        "precision": p50,
        "recall": r50,
        "f1": f1_50,
        "map50": float(stats[1]),
        "map50_95": float(stats[0]),
        "ap75": float(stats[2]),
        "matched_mean_iou": matched_mean_iou(gt_json, pred_json),
    }


def validate_config() -> None:
    missing = [regime for regime, data_yaml in REGIME_DATA_YAMLS.items() if not data_yaml]
    if missing:
        raise SystemExit(
            "Fill in REGIME_DATA_YAMLS in detector_family_comparison/comparison_config.py "
            f"before running standardized_test_eval.py. Missing: {', '.join(missing)}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate detector families on a shared split using a common COCO-style pipeline.",
    )
    parser.add_argument("--split", choices=["test", "val"], default="val", help="Dataset split to evaluate.")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for YOLO inference.")
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold for YOLO predictions.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size for YOLO prediction chunks.")
    parser.add_argument("--device", default=None, help="Torch device for YOLO inference, for example 'cpu' or '0'.")
    parser.add_argument(
        "--detectors",
        nargs="+",
        default=DETECTOR_ORDER,
        choices=DETECTOR_ORDER,
        help="Detector families to include in the shared evaluation.",
    )
    parser.add_argument(
        "--regimes",
        nargs="+",
        default=REGIME_ORDER,
        choices=REGIME_ORDER,
        help="Regimes to include in the shared evaluation.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute the selected detector/regime pairs even if they already exist in the summary CSV.",
    )
    return parser.parse_args()


def load_existing_summary(summary_csv: Path) -> list[dict[str, float | str]]:
    if not summary_csv.exists():
        return []

    rows: list[dict[str, float | str]] = []
    with summary_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed: dict[str, float | str] = {
                "detector": row["detector"],
                "regime": row["regime"],
            }
            for field in SUMMARY_FIELDS[2:]:
                parsed[field] = float(row[field])
            rows.append(parsed)
    return rows


def save_summary_csv(summary_rows: list[dict[str, float | str]], summary_csv: Path) -> None:
    ordered_rows = sorted(
        summary_rows,
        key=lambda row: (REGIME_ORDER.index(str(row["regime"])), DETECTOR_ORDER.index(str(row["detector"]))),
    )
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(ordered_rows)


def plot_standardized_summary(
    rows: list[dict[str, float | str]],
    output_path: Path,
    detector_order: list[str],
    regime_order: list[str],
    split: str,
) -> None:
    metrics = [
        ("precision", "Precision @ IoU=0.50"),
        ("recall", "Recall @ IoU=0.50"),
        ("f1", "F1 @ IoU=0.50"),
        ("map50", "mAP50"),
        ("map50_95", "mAP50-95"),
        ("matched_mean_iou", "Matched Mean IoU"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True)
    axes = axes.flatten()

    x = np.arange(len(regime_order))
    for ax, (metric, title) in zip(axes, metrics):
        for detector in detector_order:
            values = [
                next(
                    float(row[metric])
                    for row in rows
                    if row["detector"] == detector and row["regime"] == regime
                )
                for regime in regime_order
            ]
            ax.plot(x, values, marker="o", linewidth=2, label=detector)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(regime_order)
        ax.set_ylim(0, 1)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    axes[0].set_ylabel("Score")
    axes[3].set_ylabel("Score")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=max(1, len(detector_order)))
    fig.suptitle(f"Standardized detector-family comparison on {split} split", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    validate_config()
    detector_order = list(args.detectors)
    regime_order = list(args.regimes)
    output_dir = DEFAULT_OUTPUT_DIR / f"standardized_{args.split}_eval"
    gt_dir = output_dir / "ground_truth"
    pred_dir = output_dir / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = output_dir / "standardized_test_summary.csv"
    summary_rows_by_key = {
        (str(row["detector"]), str(row["regime"])): row for row in load_existing_summary(summary_csv)
    }
    selected_pairs = {(detector, regime) for detector in detector_order for regime in regime_order}
    completed_pairs = set() if args.overwrite else {key for key in summary_rows_by_key if key in selected_pairs}

    for regime in regime_order:
        data_yaml = Path(REGIME_DATA_YAMLS[regime]).resolve()
        official_gt = resolve_official_coco_gt(data_yaml, args.split)
        if official_gt is not None:
            gt_json = official_gt
            image_paths, image_id_map = load_paths_from_official_coco_gt(gt_json, data_yaml, split=args.split)
        else:
            gt_json = gt_dir / f"{regime}_{args.split}_gt.json"
            image_paths, image_id_map = load_or_build_coco_gt(data_yaml, gt_json, split=args.split)

        for detector in detector_order:
            if (detector, regime) in completed_pairs:
                print(f"Skipping completed evaluation for {detector} on {regime}.")
                continue

            if args.overwrite and (detector, regime) in summary_rows_by_key:
                print(f"Overwriting existing evaluation for {detector} on {regime}.")

            run_dir = Path(MODEL_RUNS[detector][regime])
            if detector.startswith("YOLO"):
                weights_path = run_dir / "weights" / "best.pt"
                candidate_paths = cached_prediction_candidates(pred_dir, detector, regime, args.split)
                pred_json = next((path for path in candidate_paths if path.exists()), candidate_paths[0])
                if pred_json.exists():
                    print(f"Reusing cached predictions for {detector} on {regime}: {pred_json.name}")
                else:
                    print(f"Generating predictions for {detector} on {regime}...")
                    predict_yolo_to_coco_json(
                        weights_path,
                        image_paths,
                        image_id_map,
                        pred_json,
                        imgsz=args.imgsz,
                        conf=args.conf,
                        batch=args.batch,
                        device=args.device,
                    )
            else:
                if args.split == "val":
                    run_dir = resolve_frcnn_run_dir(run_dir)
                    pred_json = run_dir / "inference" / "coco_instances_results.json"
                else:
                    candidate_paths = cached_prediction_candidates(pred_dir, detector, regime, args.split)
                    pred_json = next((path for path in candidate_paths if path.exists()), candidate_paths[0])
                if not pred_json.exists():
                    raise FileNotFoundError(
                        f"Missing Faster R-CNN predictions for split '{args.split}': {pred_json}"
                    )

            print(f"Evaluating {detector} on {regime}...")
            metrics = evaluate_coco(gt_json, pred_json)
            row = {"detector": detector, "regime": regime, **metrics}
            summary_rows_by_key[(detector, regime)] = row
            completed_pairs.add((detector, regime))
            save_summary_csv(list(summary_rows_by_key.values()), summary_csv)

    plot_detector_order = [
        detector
        for detector in DETECTOR_ORDER
        if any((detector, regime) in summary_rows_by_key for regime in regime_order)
    ]
    summary_rows = [
        summary_rows_by_key[(detector, regime)]
        for regime in regime_order
        for detector in plot_detector_order
        if (detector, regime) in summary_rows_by_key
    ]
    plot_standardized_summary(
        summary_rows,
        output_dir / "standardized_test_summary.png",
        detector_order=plot_detector_order,
        regime_order=regime_order,
        split=args.split,
    )
    print(f"Saved standardized comparison to: {output_dir}")


if __name__ == "__main__":
    main()
