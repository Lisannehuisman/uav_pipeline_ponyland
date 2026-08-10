from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean

from matched_control_lib import (
    DEFAULT_CONTROL_EXPERIMENT_ROOT,
    DEFAULT_FULL_BASELINE_SUMMARY,
    control_manifest_path,
    control_status_path,
    ensure_control_experiment_root,
    load_baseline_summary_row,
    load_control_jobs,
    read_json,
    write_csv_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate equal-image-count full-M4 control jobs into a master table and short report.",
    )
    parser.add_argument("--experiment-root", default=str(DEFAULT_CONTROL_EXPERIMENT_ROOT))
    parser.add_argument("--controls-csv", default="", help="Optional explicit matched-control manifest path.")
    parser.add_argument("--baseline-summary-csv", default=str(DEFAULT_FULL_BASELINE_SUMMARY))
    return parser.parse_args()


def float_or_nan(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def load_status_rows(experiment_root: Path, controls_csv: Path, baseline_csv: Path):
    jobs = load_control_jobs(controls_csv)
    baseline_row = load_baseline_summary_row(baseline_csv)

    rows: list[dict[str, object]] = []
    for job in jobs:
        status_path = control_status_path(experiment_root, job)
        if status_path.exists():
            status = read_json(status_path)
        else:
            status = {
                "subset": {"status": "pending"},
                "training": {"status": "pending"},
                "evaluation": {"status": "pending"},
            }

        subset = status.get("subset", {})
        training = status.get("training", {})
        evaluation = status.get("evaluation", {})
        metrics = evaluation.get("metrics", {})
        split_counts = subset.get("split_counts", {})

        row: dict[str, object] = {
            "control_id": job.control_id,
            "label": job.label,
            "source_group": job.source_group,
            "source_id": job.source_id,
            "source_label": job.source_label,
            "seed": job.seed,
            "sampling_strategy": job.sampling_strategy,
            "number_of_train_images": split_counts.get("train_images", job.train_count),
            "number_of_val_images": split_counts.get("val_images", job.val_count),
            "number_of_test_images": metrics.get("num_test_images", split_counts.get("test_full_images", "")),
            "precision": float_or_nan(metrics.get("precision")),
            "recall": float_or_nan(metrics.get("recall")),
            "F1": float_or_nan(metrics.get("f1")),
            "mAP50": float_or_nan(metrics.get("map50")),
            "mAP50-95": float_or_nan(metrics.get("map50_95")),
            "training_status": training.get("status", "pending"),
            "model_path": training.get("model_path", ""),
            "subset_status": subset.get("status", "pending"),
            "evaluation_status": evaluation.get("status", "pending"),
            "reference_mAP50-95": job.reference_map50_95,
            "reference_mAP50": job.reference_map50,
            "reference_F1": job.reference_f1,
            "delta_mAP50-95_vs_reference": float_or_nan(metrics.get("delta_map50_95_vs_reference")),
            "delta_mAP50_vs_reference": float_or_nan(metrics.get("delta_map50_vs_reference")),
            "delta_F1_vs_reference": float_or_nan(metrics.get("delta_f1_vs_reference")),
        }
        if baseline_row is not None:
            row["delta_mAP50-95_vs_full_M4"] = float_or_nan(metrics.get("delta_map50_95_vs_full_m4"))
            row["delta_mAP50_vs_full_M4"] = float_or_nan(metrics.get("delta_map50_vs_full_m4"))
            row["delta_F1_vs_full_M4"] = float_or_nan(metrics.get("delta_f1_vs_full_m4"))
        else:
            row["delta_mAP50-95_vs_full_M4"] = float("nan")
            row["delta_mAP50_vs_full_M4"] = float("nan")
            row["delta_F1_vs_full_M4"] = float("nan")
        rows.append(row)

    return rows, baseline_row


def completed_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [row for row in rows if row.get("evaluation_status") == "completed"]


def write_report(rows: list[dict[str, object]], output_path: Path) -> None:
    completed = completed_rows(rows)
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in completed:
        grouped.setdefault((str(row["source_group"]), str(row["source_id"])), []).append(row)

    lines = [
        "# M4 Matched-Control Report",
        "",
        "## What This Experiment Does",
        "",
        "- It trains full-M4 controls with the same train/val image counts as the source single-view or pair-view model.",
        "- The control images are sampled from the full M4 viewpoint space, so this isolates `image count` from `viewpoint restriction`.",
        "- Each control is evaluated on the same fixed full M4 test split as the source models.",
        "",
        "## Current Status",
        "",
        f"- Defined controls: {len(rows)}",
        f"- Completed controls: {len(completed)}",
        "",
        "## Source-Level Summary",
        "",
    ]

    if not grouped:
        lines.append("- No completed matched controls yet.")
    else:
        for (source_group, source_id), source_rows in sorted(grouped.items()):
            source_label = str(source_rows[0]["source_label"])
            mean_map = mean(float(row["mAP50-95"]) for row in source_rows)
            mean_gap = mean(float(row["delta_mAP50-95_vs_reference"]) for row in source_rows)
            train_count = int(source_rows[0]["number_of_train_images"])
            val_count = int(source_rows[0]["number_of_val_images"])
            lines.append(
                f"- `{source_group}` `{source_id}` (`{source_label}`): "
                f"{len(source_rows)} control run(s), "
                f"`train={train_count}`, `val={val_count}`, "
                f"mean control `mAP50-95 = {mean_map:.4f}`, "
                f"mean gap vs source `{mean_gap:+.4f}`"
            )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    experiment_root = Path(args.experiment_root).resolve()
    ensure_control_experiment_root(experiment_root)
    controls_csv = Path(args.controls_csv).resolve() if args.controls_csv else control_manifest_path(experiment_root)
    baseline_csv = Path(args.baseline_summary_csv).resolve()

    rows, _ = load_status_rows(experiment_root, controls_csv, baseline_csv)
    reports_dir = experiment_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    write_csv_rows(
        reports_dir / "master_results.csv",
        [
            "control_id",
            "label",
            "source_group",
            "source_id",
            "source_label",
            "seed",
            "sampling_strategy",
            "number_of_train_images",
            "number_of_val_images",
            "number_of_test_images",
            "precision",
            "recall",
            "F1",
            "mAP50",
            "mAP50-95",
            "training_status",
            "model_path",
            "subset_status",
            "evaluation_status",
            "reference_mAP50-95",
            "reference_mAP50",
            "reference_F1",
            "delta_mAP50-95_vs_reference",
            "delta_mAP50_vs_reference",
            "delta_F1_vs_reference",
            "delta_mAP50-95_vs_full_M4",
            "delta_mAP50_vs_full_M4",
            "delta_F1_vs_full_M4",
        ],
        rows,
    )
    write_report(rows, reports_dir / "matched_control_report.md")
    print(f"Wrote matched-control outputs under: {reports_dir}")


if __name__ == "__main__":
    main()
