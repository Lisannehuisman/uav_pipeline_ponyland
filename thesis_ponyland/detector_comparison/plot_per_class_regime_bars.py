from __future__ import annotations

import argparse
import contextlib
import csv
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from comparison_config import DEFAULT_OUTPUT_DIR, DETECTOR_ORDER, MODEL_RUNS, REGIME_DATA_YAMLS, REGIME_ORDER


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot per-class AP across regimes from standardized prediction JSONs.",
    )
    parser.add_argument("--split", choices=["val", "test"], default="val", help="Shared split to use.")
    parser.add_argument(
        "--metric",
        choices=["ap50_95", "ap50", "ap75"],
        default="ap50_95",
        help="Per-class metric to visualize.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for the CSV and PNG outputs.",
    )
    return parser.parse_args()


def resolve_frcnn_run_dir(run_dir: Path) -> Path:
    if (run_dir / "inference" / "coco_instances_results.json").exists():
        return run_dir
    nested_candidates = list(run_dir.glob("*/inference/coco_instances_results.json"))
    if nested_candidates:
        return nested_candidates[0].parent.parent
    return run_dir


def resolve_official_coco_gt(data_yaml: Path, split: str) -> Path:
    if split == "test":
        candidate = DEFAULT_OUTPUT_DIR / "standardized_test_eval" / "ground_truth" / f"{data_yaml.stem}_{split}_gt.json"
        if candidate.exists():
            return candidate
        raise FileNotFoundError(
            f"Missing cached test ground-truth JSON for {data_yaml.stem}: {candidate}. "
            "Run standardized_test_eval.py on the test split first."
        )

    with data_yaml.open("r", encoding="utf-8") as handle:
        data_dict = yaml.safe_load(handle)

    dataset_root = Path(data_dict["path"])
    if not dataset_root.is_absolute():
        dataset_root = (data_yaml.parent / dataset_root).resolve()

    regime_name = data_yaml.stem
    candidates = [
        dataset_root / "coco_annotations" / f"coco_instances_{split}_{regime_name}.json",
        dataset_root / "annotations" / f"instances_{split}_{regime_name}.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find a COCO annotation JSON for {split} in {data_yaml}.")


def yolo_prediction_json(detector: str, regime: str, split: str) -> Path:
    prediction_dir = DEFAULT_OUTPUT_DIR / f"standardized_{split}_eval" / "predictions"
    candidates = [
        prediction_dir / f"{detector.replace(' ', '_')}_{regime}_{split}_predictions.json",
        prediction_dir / f"{detector.replace(' ', '_')}_{regime}_predictions.json",
    ]
    return next((path for path in candidates if path.exists()), candidates[0])


def detector_prediction_json(detector: str, regime: str, split: str) -> Path:
    if detector.startswith("YOLO"):
        prediction_path = yolo_prediction_json(detector, regime, split)
    else:
        if split == "val":
            run_dir = resolve_frcnn_run_dir(Path(MODEL_RUNS[detector][regime]))
            prediction_path = run_dir / "inference" / "coco_instances_results.json"
        else:
            prediction_dir = DEFAULT_OUTPUT_DIR / f"standardized_{split}_eval" / "predictions"
            candidates = [
                prediction_dir / f"{detector.replace(' ', '_')}_{regime}_{split}_predictions.json",
                prediction_dir / f"{detector.replace(' ', '_')}_{regime}_predictions.json",
            ]
            prediction_path = next((path for path in candidates if path.exists()), candidates[0])

    if not prediction_path.exists():
        raise FileNotFoundError(f"Missing prediction JSON for {detector} / {regime}: {prediction_path}")
    return prediction_path


def category_names_from_gt(gt_json: Path) -> list[str]:
    gt = COCO(str(gt_json))
    categories = [gt.cats[cat_id]["name"] for cat_id in sorted(gt.cats)]
    return categories


def per_class_ap(gt_json: Path, pred_json: Path, metric: str) -> dict[str, float]:
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(str(gt_json))
        coco_dt = coco_gt.loadRes(str(pred_json))
        evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
        evaluator.evaluate()
        evaluator.accumulate()

    precision = evaluator.eval["precision"]
    if metric == "ap50":
        iou_values = evaluator.params.iouThrs
        precision = precision[[int(np.argmin(np.abs(iou_values - 0.5)))]]
    elif metric == "ap75":
        iou_values = evaluator.params.iouThrs
        precision = precision[[int(np.argmin(np.abs(iou_values - 0.75)))]]

    results: dict[str, float] = {}
    for class_index, cat_id in enumerate(sorted(coco_gt.cats)):
        class_name = coco_gt.cats[cat_id]["name"]
        class_precision = precision[:, :, class_index, 0, -1]
        valid = class_precision[class_precision > -1]
        results[class_name] = float(np.mean(valid)) if valid.size else float("nan")
    return results


def save_per_class_csv(rows: list[dict[str, str | float]], output_csv: Path) -> None:
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["regime", "detector", "class_name", "metric", "score"])
        writer.writeheader()
        writer.writerows(rows)


def plot_regime_grid(
    rows: list[dict[str, str | float]],
    class_names: list[str],
    metric: str,
    output_png: Path,
) -> None:
    lookup = {(str(row["regime"]), str(row["detector"]), str(row["class_name"])): float(row["score"]) for row in rows}
    colors = {
        "YOLOv8n": "#4C78A8",
        "YOLOv8l": "#F58518",
        "Faster R-CNN": "#54A24B",
    }
    metric_label = {"ap50_95": "AP50-95", "ap50": "AP50", "ap75": "AP75"}[metric]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)
    axes = axes.flatten()
    x = np.arange(len(class_names))
    width = 0.25

    for index, regime in enumerate(REGIME_ORDER):
        ax = axes[index]
        for detector_idx, detector in enumerate(DETECTOR_ORDER):
            values = [lookup[(regime, detector, class_name)] for class_name in class_names]
            offset = (detector_idx - 1) * width
            ax.bar(x + offset, values, width=width, label=detector if index == 0 else None, color=colors[detector])

        ax.set_title(regime)
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=40, ha="right")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        if index % 3 == 0:
            ax.set_ylabel(metric_label)

    for axis in axes[len(REGIME_ORDER) :]:
        axis.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.04, 0.90))
    fig.suptitle(
        f"Per-class {metric_label} across regimes: YOLOv8n vs YOLOv8l vs Faster R-CNN",
        fontsize=17,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    default_output_dir = DEFAULT_OUTPUT_DIR / f"standardized_{args.split}_eval" / "per_class_plots"
    output_dir = Path(args.output_dir).resolve() if args.output_dir else default_output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_gt = resolve_official_coco_gt(Path(REGIME_DATA_YAMLS[REGIME_ORDER[0]]), args.split)
    class_names = category_names_from_gt(reference_gt)

    rows: list[dict[str, str | float]] = []
    for regime in REGIME_ORDER:
        gt_json = resolve_official_coco_gt(Path(REGIME_DATA_YAMLS[regime]), args.split)
        for detector in DETECTOR_ORDER:
            pred_json = detector_prediction_json(detector, regime, args.split)
            scores = per_class_ap(gt_json, pred_json, metric=args.metric)
            for class_name in class_names:
                rows.append(
                    {
                        "regime": regime,
                        "detector": detector,
                        "class_name": class_name,
                        "metric": args.metric,
                        "score": scores[class_name],
                    }
                )

    output_csv = output_dir / f"per_class_{args.metric}_{args.split}.csv"
    output_png = output_dir / f"per_class_{args.metric}_{args.split}.png"
    save_per_class_csv(rows, output_csv)
    plot_regime_grid(rows, class_names, args.metric, output_png)
    print(f"Saved per-class CSV to: {output_csv}")
    print(f"Saved per-class figure to: {output_png}")


if __name__ == "__main__":
    main()
