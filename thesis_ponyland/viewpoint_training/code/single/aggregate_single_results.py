from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from single_experiment_lib import (
    DEFAULT_FULL_BASELINE_SUMMARY,
    ensure_single_experiment_root,
    human_viewpoint_label,
    load_baseline_summary_row,
    load_single_jobs,
    read_json,
    short_viewpoint_label,
    single_manifest_path,
    single_status_path,
    write_csv_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate M4 single-viewpoint experiment status files into a master table, plots, and report.",
    )
    parser.add_argument("--experiment-root", default="outputs/m4_single_subset_experiment")
    parser.add_argument("--singles-csv", default="", help="Optional explicit single-viewpoint manifest path.")
    parser.add_argument("--baseline-summary-csv", default=str(DEFAULT_FULL_BASELINE_SUMMARY))
    parser.add_argument("--top-k", type=int, default=15)
    return parser.parse_args()


def float_or_nan(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def load_status_rows(experiment_root: Path, singles_csv: Path, baseline_csv: Path):
    jobs = load_single_jobs(singles_csv)
    baseline_row = load_baseline_summary_row(baseline_csv)

    rows: list[dict[str, object]] = []
    for job in jobs:
        status_path = single_status_path(experiment_root, job)
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
            "single_id": job.single_id,
            "viewpoint": job.viewpoint,
            "number_of_train_images": split_counts.get("train_images", ""),
            "number_of_val_images": split_counts.get("val_images", ""),
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
            "pilot_rank": "" if job.pilot_rank is None else job.pilot_rank,
            "pilot_name": job.pilot_name,
            "pilot_note": job.pilot_note,
        }
        if baseline_row is not None:
            row["delta_mAP50-95_vs_full_M4"] = row["mAP50-95"] - float_or_nan(baseline_row.get("map50_95"))
            row["delta_mAP50_vs_full_M4"] = row["mAP50"] - float_or_nan(baseline_row.get("map50"))
        else:
            row["delta_mAP50-95_vs_full_M4"] = float("nan")
            row["delta_mAP50_vs_full_M4"] = float("nan")
        rows.append(row)

    return rows, baseline_row


def completed_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        row
        for row in rows
        if row.get("evaluation_status") == "completed" and not math.isnan(float_or_nan(row.get("mAP50-95")))
    ]


def save_master_tables(rows: list[dict[str, object]], experiment_root: Path) -> None:
    output_dir = experiment_root / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    fields = [
        "single_id",
        "viewpoint",
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
        "delta_mAP50-95_vs_full_M4",
        "delta_mAP50_vs_full_M4",
        "pilot_rank",
        "pilot_name",
        "pilot_note",
    ]
    write_csv_rows(output_dir / "master_results.csv", fields, rows)


def plot_top_viewpoints(rows: list[dict[str, object]], output_path: Path, top_k: int) -> None:
    completed = sorted(completed_rows(rows), key=lambda row: float(row["mAP50-95"]), reverse=True)[:top_k]
    fig, ax = plt.subplots(figsize=(14, max(6, len(completed) * 0.4)))
    if not completed:
        ax.text(0.5, 0.5, "No completed single-viewpoint evaluations yet.", ha="center", va="center")
        ax.axis("off")
    else:
        labels = [f"{row['single_id']}: {short_viewpoint_label(str(row['viewpoint']))}" for row in completed]
        values = [float(row["mAP50-95"]) for row in completed]
        ax.barh(labels[::-1], values[::-1], color="#1f77b4")
        ax.set_xlabel("mAP50-95 on full fixed M4 test")
        ax.set_title("Top-performing training viewpoints")
        ax.grid(axis="x", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_distribution(rows: list[dict[str, object]], output_path: Path) -> None:
    completed = completed_rows(rows)
    fig, ax = plt.subplots(figsize=(10, 6))
    if not completed:
        ax.text(0.5, 0.5, "No completed single-viewpoint evaluations yet.", ha="center", va="center")
        ax.axis("off")
    else:
        values = [float(row["mAP50-95"]) for row in completed]
        ax.hist(values, bins=min(18, max(8, len(values) // 4)), color="#ff7f0e", edgecolor="white", alpha=0.9)
        ax.set_xlabel("mAP50-95 on full fixed M4 test")
        ax.set_ylabel("Number of viewpoints")
        ax.set_title("Distribution of single-viewpoint training performance")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_viewpoint_heatmap(rows: list[dict[str, object]], output_path: Path) -> None:
    completed = completed_rows(rows)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    elevation_order = ["ellow", "elmid", "elhigh"]
    radius_order = ["radnear", "radmid", "radfar"]
    azimuth_order = [0, 45, 90, 135, 180, 225, 270, 315]

    lookup = {str(row["viewpoint"]): float(row["mAP50-95"]) for row in completed}
    image = None
    any_data = False
    for ax, elevation in zip(axes, elevation_order, strict=True):
        grid = np.full((len(radius_order), len(azimuth_order)), np.nan, dtype=float)
        for r_index, radius in enumerate(radius_order):
            for a_index, azimuth in enumerate(azimuth_order):
                viewpoint = f"{elevation}-{radius}-az{azimuth:03d}"
                if viewpoint in lookup:
                    grid[r_index, a_index] = lookup[viewpoint]
                    any_data = True
        if np.isnan(grid).all():
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.axis("off")
            continue
        image = ax.imshow(grid, cmap="viridis", interpolation="nearest", aspect="auto")
        ax.set_title(f"{elevation}")
        ax.set_xticks(np.arange(len(azimuth_order)))
        ax.set_xticklabels([f"{az:03d}" for az in azimuth_order], rotation=45, ha="right")
        ax.set_yticks(np.arange(len(radius_order)))
        ax.set_yticklabels(radius_order)
        ax.set_xlabel("Azimuth")
        if ax is axes[0]:
            ax.set_ylabel("Radius")
    if any_data and image is not None:
        cbar = fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.03, pad=0.03)
        cbar.set_label("mAP50-95")
    fig.suptitle("Training-viewpoint performance heatmap by elevation, radius, and azimuth")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_gain_loss(rows: list[dict[str, object]], baseline_row, output_path: Path) -> None:
    completed = completed_rows(rows)
    fig, ax = plt.subplots(figsize=(10, 6))
    if not completed or baseline_row is None:
        ax.text(0.5, 0.5, "Baseline or completed single-viewpoint results unavailable.", ha="center", va="center")
        ax.axis("off")
    else:
        values = [float(row["delta_mAP50-95_vs_full_M4"]) for row in completed]
        ax.hist(values, bins=min(18, max(8, len(values) // 4)), color="#2ca02c", edgecolor="white", alpha=0.9)
        ax.axvline(0.0, color="#d62728", linestyle="--", linewidth=1.5)
        ax.set_xlabel("Delta mAP50-95 vs full M4 YOLOv8l baseline")
        ax.set_ylabel("Number of viewpoints")
        ax.set_title("Gain/loss relative to the full M4 model")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_report(rows: list[dict[str, object]], baseline_row, output_path: Path) -> None:
    total_views = len(rows)
    completed = completed_rows(rows)
    pilot_rows = sorted(
        [row for row in rows if str(row.get("pilot_rank", "")).strip()],
        key=lambda row: int(row["pilot_rank"]),
    )

    lines = [
        "# M4 Single-Viewpoint Training Report",
        "",
        "## What Was Trained",
        "",
        "- One normal single-image YOLOv8l detector per training viewpoint.",
        "- Each model is trained only on M4 `train` images whose filenames match the selected viewpoint.",
        "- Validation during training uses the matching viewpoint-filtered `val` split.",
        "- Labels are preserved exactly by reusing the original M4 label files through YOLO list files.",
        "",
        "## Scientific Question",
        "",
        "- This experiment asks how much detector generalization can be learned from training on only one viewpoint.",
        "- It provides a matched single-view baseline for the duo-viewpoint training sweep.",
        "",
        "## Evaluation Protocol",
        "",
        "- Every viewpoint-trained model is evaluated on the full fixed M4 test split across all 72 viewpoints.",
        "- This measures generalization from a restricted training subset to the full viewpoint space.",
        "",
        "## Current Sweep Status",
        "",
        f"- Viewpoint definitions: {total_views}",
        f"- Completed evaluations: {len(completed)}",
    ]

    if baseline_row is not None:
        lines.extend(
            [
                "",
                "## Full M4 Baseline",
                "",
                f"- Full M4 YOLOv8l baseline `mAP50-95`: {float(baseline_row['map50_95']):.4f}",
                f"- Full M4 YOLOv8l baseline `mAP50`: {float(baseline_row['map50']):.4f}",
                f"- Full M4 YOLOv8l baseline `F1`: {float(baseline_row['f1']):.4f}",
            ]
        )

    if completed:
        best_row = sorted(completed, key=lambda row: float(row["mAP50-95"]), reverse=True)[0]
        worst_row = sorted(completed, key=lambda row: float(row["mAP50-95"]))[0]
        lines.extend(
            [
                "",
                "## Best Completed Training Viewpoint",
                "",
                f"- Viewpoint id: `{best_row['single_id']}`",
                f"- Viewpoint: `{best_row['viewpoint']}` ({human_viewpoint_label(str(best_row['viewpoint']))})",
                f"- `mAP50-95`: {float(best_row['mAP50-95']):.4f}",
                f"- `mAP50`: {float(best_row['mAP50']):.4f}",
                f"- `F1`: {float(best_row['F1']):.4f}",
                "",
                "## Worst Completed Training Viewpoint",
                "",
                f"- Viewpoint id: `{worst_row['single_id']}`",
                f"- Viewpoint: `{worst_row['viewpoint']}` ({human_viewpoint_label(str(worst_row['viewpoint']))})",
                f"- `mAP50-95`: {float(worst_row['mAP50-95']):.4f}",
                f"- `mAP50`: {float(worst_row['mAP50']):.4f}",
                f"- `F1`: {float(worst_row['F1']):.4f}",
            ]
        )

    if pilot_rows:
        lines.extend(["", "## Pilot Viewpoints", ""])
        for row in pilot_rows:
            lines.append(
                f"- Pilot {int(row['pilot_rank'])}: `{row['single_id']}` = `{row['viewpoint']}` ({row['pilot_name']})"
            )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    experiment_root = Path(args.experiment_root).resolve()
    ensure_single_experiment_root(experiment_root)
    singles_csv = Path(args.singles_csv).resolve() if args.singles_csv else single_manifest_path(experiment_root)
    baseline_csv = Path(args.baseline_summary_csv).resolve()

    rows, baseline_row = load_status_rows(experiment_root, singles_csv, baseline_csv)
    save_master_tables(rows, experiment_root)

    plots_dir = experiment_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_top_viewpoints(rows, plots_dir / "top_performing_single_viewpoints.png", top_k=args.top_k)
    plot_distribution(rows, plots_dir / "single_viewpoint_performance_distribution.png")
    plot_viewpoint_heatmap(rows, plots_dir / "single_viewpoint_performance_heatmap.png")
    plot_gain_loss(rows, baseline_row, plots_dir / "gain_loss_vs_full_m4.png")

    report_path = experiment_root / "reports" / "single_viewpoint_experiment_report.md"
    write_report(rows, baseline_row, report_path)
    print(f"Wrote master tables, plots, and report under: {experiment_root}")


if __name__ == "__main__":
    main()
