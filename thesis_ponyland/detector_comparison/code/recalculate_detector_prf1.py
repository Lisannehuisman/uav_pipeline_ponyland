#!/usr/bin/env python3

import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

PROJECT = Path(
    "/vol/tensusers6/lisannehuisman/projects/compare_yolov8l_frcnn"
)
MODULE_DIR = PROJECT / "detector_family_comparison"

sys.path.insert(0, str(MODULE_DIR))

from comparison_config import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    DETECTOR_ORDER,
    MODEL_RUNS,
    REGIME_DATA_YAMLS,
    REGIME_ORDER,
)

from standardized_test_eval import (  # noqa: E402
    cached_prediction_candidates,
    load_or_build_coco_gt,
    resolve_frcnn_run_dir,
    resolve_official_coco_gt,
)


IOU_THRESHOLD = 0.50
MAX_DETECTIONS_PER_IMAGE = 100

OUTPUT_ROOT = Path(DEFAULT_OUTPUT_DIR)
CORRECTED_DIR = OUTPUT_ROOT / "corrected_prf1_20260720"
CORRECTED_DIR.mkdir(parents=True, exist_ok=True)

GT_CACHE_DIR = CORRECTED_DIR / "ground_truth"
GT_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def bbox_iou_xywh(box_a, box_b):
    """IoU for COCO-format boxes [x, y, width, height]."""

    ax, ay, aw, ah = map(float, box_a)
    bx, by, bw, bh = map(float, box_b)

    ax2 = ax + aw
    ay2 = ay + ah
    bx2 = bx + bw
    by2 = by + bh

    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)

    intersection = iw * ih
    area_a = max(0.0, aw) * max(0.0, ah)
    area_b = max(0.0, bw) * max(0.0, bh)

    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def get_ground_truth(regime, split):
    """
    Use exactly the same GT-resolution logic as the standardized evaluator.
    """

    data_yaml = Path(REGIME_DATA_YAMLS[regime]).resolve()

    official_gt = resolve_official_coco_gt(data_yaml, split)

    if official_gt is not None:
        return Path(official_gt)

    gt_json = GT_CACHE_DIR / f"{regime}_{split}_gt.json"

    load_or_build_coco_gt(
        data_yaml,
        gt_json,
        split=split,
    )

    return gt_json


def get_prediction_file(detector, regime, split):
    """
    Resolve the same prediction files used by standardized_test_eval.py.
    """

    run_dir = Path(MODEL_RUNS[detector][regime])

    # YOLO validation/test predictions are cached in standardized_*_eval.
    if detector.startswith("YOLO"):
        pred_dir = (
            OUTPUT_ROOT
            / f"standardized_{split}_eval"
            / "predictions"
        )

        candidates = cached_prediction_candidates(
            pred_dir,
            detector,
            regime,
            split,
        )

        for path in candidates:
            if path.exists():
                return path

        raise FileNotFoundError(
            f"No {split} prediction JSON found for "
            f"{detector} {regime}.\nCandidates:\n"
            + "\n".join(str(p) for p in candidates)
        )

    # Faster R-CNN validation predictions are stored in each run directory.
    if split == "val":
        resolved_run = resolve_frcnn_run_dir(run_dir)

        path = (
            resolved_run
            / "inference"
            / "coco_instances_results.json"
        )

        if not path.exists():
            raise FileNotFoundError(path)

        return path

    # Faster R-CNN test predictions were standardized/cached earlier.
    pred_dir = (
        OUTPUT_ROOT
        / "standardized_test_eval"
        / "predictions"
    )

    candidates = cached_prediction_candidates(
        pred_dir,
        detector,
        regime,
        split,
    )

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"No test prediction JSON found for "
        f"{detector} {regime}.\nCandidates:\n"
        + "\n".join(str(p) for p in candidates)
    )


def prepare_detection_labels(gt_path, pred_path):
    """
    Match detections to GT once, in descending confidence order.

    Matching rules:
    - same image
    - same category
    - IoU >= 0.50
    - each GT matched at most once
    - maximum 100 detections per image, consistent with COCO maxDets=100

    Returns:
        labels: [(score, TP, FP), ...] sorted by descending score
        total_gt: number of non-crowd GT objects
    """

    gt_data = json.loads(
        Path(gt_path).read_text(encoding="utf-8")
    )

    predictions = json.loads(
        Path(pred_path).read_text(encoding="utf-8")
    )

    gt_by_key = defaultdict(list)
    total_gt = 0

    for ann in gt_data["annotations"]:

        # Crowd boxes are not treated as ordinary GT targets.
        if int(ann.get("iscrowd", 0)) == 1:
            continue

        key = (
            int(ann["image_id"]),
            int(ann["category_id"]),
        )

        gt_by_key[key].append(
            {
                "bbox": ann["bbox"],
                "matched": False,
            }
        )

        total_gt += 1

    # COCO uses maxDets=100.
    # Keep top 100 detections per image before global evaluation.
    preds_by_image = defaultdict(list)

    for pred in predictions:
        preds_by_image[int(pred["image_id"])].append(pred)

    kept_predictions = []

    for image_id, image_preds in preds_by_image.items():

        image_preds = sorted(
            image_preds,
            key=lambda p: float(p["score"]),
            reverse=True,
        )

        kept_predictions.extend(
            image_preds[:MAX_DETECTIONS_PER_IMAGE]
        )

    # Global confidence ordering.
    kept_predictions.sort(
        key=lambda p: float(p["score"]),
        reverse=True,
    )

    labels = []

    for pred in kept_predictions:

        score = float(pred["score"])

        key = (
            int(pred["image_id"]),
            int(pred["category_id"]),
        )

        candidates = gt_by_key.get(key, [])

        best_iou = -1.0
        best_index = None

        for index, gt in enumerate(candidates):

            if gt["matched"]:
                continue

            iou = bbox_iou_xywh(
                pred["bbox"],
                gt["bbox"],
            )

            if iou > best_iou:
                best_iou = iou
                best_index = index

        if (
            best_index is not None
            and best_iou >= IOU_THRESHOLD
        ):
            candidates[best_index]["matched"] = True
            labels.append((score, 1, 0))
        else:
            labels.append((score, 0, 1))

    return labels, total_gt


def metrics_from_counts(tp, fp, total_gt):

    fn = total_gt - tp

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / total_gt
        if total_gt > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def find_best_validation_threshold(labels, total_gt):
    """
    Find the exact score threshold that maximizes F1 on validation.

    All predictions sharing the same score are processed together,
    so the selected operating point corresponds to score >= threshold.
    """

    if not labels:
        result = metrics_from_counts(0, 0, total_gt)
        return {
            "threshold": 1.0,
            **result,
        }

    tp = 0
    fp = 0

    best = None

    index = 0

    while index < len(labels):

        current_score = labels[index][0]

        # Include every detection with this exact score.
        while (
            index < len(labels)
            and labels[index][0] == current_score
        ):
            tp += labels[index][1]
            fp += labels[index][2]
            index += 1

        metrics = metrics_from_counts(
            tp,
            fp,
            total_gt,
        )

        candidate = {
            "threshold": float(current_score),
            **metrics,
        }

        if best is None:
            best = candidate

        elif candidate["f1"] > best["f1"] + 1e-12:
            best = candidate

        elif abs(
            candidate["f1"] - best["f1"]
        ) <= 1e-12:
            # Tie: choose the higher/more conservative threshold.
            if candidate["threshold"] > best["threshold"]:
                best = candidate

    return best


def evaluate_at_threshold(
    labels,
    total_gt,
    threshold,
):
    """
    Evaluate test predictions at the frozen validation threshold.
    """

    tp = 0
    fp = 0

    for score, is_tp, is_fp in labels:

        if score < threshold:
            break

        tp += is_tp
        fp += is_fp

    return {
        "threshold": float(threshold),
        **metrics_from_counts(
            tp,
            fp,
            total_gt,
        ),
    }


def main():

    threshold_rows = []
    test_rows = []

    print(
        "\nCorrect recalculation of Precision / Recall / F1"
    )
    print(
        f"IoU threshold = {IOU_THRESHOLD}"
    )
    print(
        f"Maximum detections per image = "
        f"{MAX_DETECTIONS_PER_IMAGE}"
    )

    for regime in REGIME_ORDER:

        print(f"\n===== {regime} =====")

        val_gt = get_ground_truth(
            regime,
            "val",
        )

        test_gt = get_ground_truth(
            regime,
            "test",
        )

        for detector in DETECTOR_ORDER:

            print(
                f"\n{detector} {regime}"
            )

            val_pred = get_prediction_file(
                detector,
                regime,
                "val",
            )

            test_pred = get_prediction_file(
                detector,
                regime,
                "test",
            )

            print(
                f"  VAL predictions:  {val_pred}"
            )

            print(
                f"  TEST predictions: {test_pred}"
            )

            val_labels, val_total_gt = (
                prepare_detection_labels(
                    val_gt,
                    val_pred,
                )
            )

            best_val = (
                find_best_validation_threshold(
                    val_labels,
                    val_total_gt,
                )
            )

            test_labels, test_total_gt = (
                prepare_detection_labels(
                    test_gt,
                    test_pred,
                )
            )

            test_result = evaluate_at_threshold(
                test_labels,
                test_total_gt,
                best_val["threshold"],
            )

            threshold_rows.append(
                {
                    "detector": detector,
                    "regime": regime,
                    "selected_conf_threshold":
                        best_val["threshold"],
                    "val_tp": best_val["tp"],
                    "val_fp": best_val["fp"],
                    "val_fn": best_val["fn"],
                    "val_precision":
                        best_val["precision"],
                    "val_recall":
                        best_val["recall"],
                    "val_f1":
                        best_val["f1"],
                }
            )

            test_rows.append(
                {
                    "detector": detector,
                    "regime": regime,
                    "selected_conf_threshold":
                        best_val["threshold"],
                    "tp": test_result["tp"],
                    "fp": test_result["fp"],
                    "fn": test_result["fn"],
                    "gt_total": test_total_gt,
                    "precision":
                        test_result["precision"],
                    "recall":
                        test_result["recall"],
                    "f1":
                        test_result["f1"],
                }
            )

            print(
                "  Selected on VAL:"
                f" threshold={best_val['threshold']:.6f}"
                f" | P={best_val['precision']:.4f}"
                f" | R={best_val['recall']:.4f}"
                f" | F1={best_val['f1']:.4f}"
            )

            print(
                "  Applied to TEST:"
                f" TP={test_result['tp']}"
                f" FP={test_result['fp']}"
                f" FN={test_result['fn']}"
                f" | P={test_result['precision']:.4f}"
                f" | R={test_result['recall']:.4f}"
                f" | F1={test_result['f1']:.4f}"
            )

    thresholds_df = pd.DataFrame(
        threshold_rows
    )

    test_df = pd.DataFrame(
        test_rows
    )

    threshold_csv = (
        CORRECTED_DIR
        / "validation_selected_thresholds.csv"
    )

    test_csv = (
        CORRECTED_DIR
        / "corrected_test_prf1.csv"
    )

    thresholds_df.to_csv(
        threshold_csv,
        index=False,
    )

    test_df.to_csv(
        test_csv,
        index=False,
    )

    # Merge corrected P/R/F1 with the already-valid COCO AP metrics.
    original_summary_path = (
        OUTPUT_ROOT
        / "standardized_test_eval"
        / "standardized_test_summary.csv"
    )

    original = pd.read_csv(
        original_summary_path
    )

    corrected_metrics = test_df[
        [
            "detector",
            "regime",
            "selected_conf_threshold",
            "tp",
            "fp",
            "fn",
            "precision",
            "recall",
            "f1",
        ]
    ]

    original_without_wrong_prf1 = (
        original.drop(
            columns=[
                "precision",
                "recall",
                "f1",
            ],
            errors="ignore",
        )
    )

    corrected_summary = (
        original_without_wrong_prf1.merge(
            corrected_metrics,
            on=[
                "detector",
                "regime",
            ],
            how="left",
        )
    )

    desired_order = [
        "detector",
        "regime",
        "selected_conf_threshold",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
        "map50",
        "map50_95",
        "ap75",
        "matched_mean_iou",
    ]

    corrected_summary = corrected_summary[
        [
            column
            for column in desired_order
            if column in corrected_summary.columns
        ]
    ]

    corrected_summary_path = (
        CORRECTED_DIR
        / "standardized_test_summary_CORRECTED.csv"
    )

    corrected_summary.to_csv(
        corrected_summary_path,
        index=False,
    )

    print("\n\n==============================")
    print("FILES WRITTEN")
    print("==============================")

    print(threshold_csv)
    print(test_csv)
    print(corrected_summary_path)

    print("\nCorrected TEST metrics:\n")

    display_cols = [
        "detector",
        "regime",
        "selected_conf_threshold",
        "precision",
        "recall",
        "f1",
    ]

    print(
        corrected_summary[
            display_cols
        ].to_string(index=False)
    )

    # Sanity checks
    print("\nSANITY CHECKS")

    formula_p = (
        test_df["tp"]
        / (
            test_df["tp"]
            + test_df["fp"]
        )
    )

    formula_r = (
        test_df["tp"]
        / (
            test_df["tp"]
            + test_df["fn"]
        )
    )

    formula_f1 = (
        2
        * formula_p
        * formula_r
        / (
            formula_p
            + formula_r
        )
    )

    p_ok = (
        abs(
            formula_p
            - test_df["precision"]
        ) < 1e-10
    ).all()

    r_ok = (
        abs(
            formula_r
            - test_df["recall"]
        ) < 1e-10
    ).all()

    f1_ok = (
        abs(
            formula_f1
            - test_df["f1"]
        ).fillna(0) < 1e-10
    ).all()

    print(
        f"Precision formulas correct: {p_ok}"
    )

    print(
        f"Recall formulas correct:    {r_ok}"
    )

    print(
        f"F1 formulas correct:        {f1_ok}"
    )

    print(
        "\nGT totals found:",
        sorted(
            test_df["gt_total"]
            .unique()
            .tolist()
        ),
    )

    same_as_map50 = (
        abs(
            corrected_summary["precision"]
            - corrected_summary["map50"]
        ) < 1e-10
    ).sum()

    print(
        "Rows where corrected precision "
        f"exactly equals mAP50: "
        f"{same_as_map50}/"
        f"{len(corrected_summary)}"
    )


if __name__ == "__main__":
    main()
