from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pair_experiment_lib import (
    DEFAULT_FULL_BASELINE_SUMMARY,
    DEFAULT_PROTOCOL_RECOMMENDATION,
    ensure_experiment_root,
    load_baseline_summary_row,
    load_pair_jobs,
    pair_manifest_path,
    pair_status_path,
    read_json,
    short_viewpoint_label,
    viewpoint_sort_key,
    write_csv_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate M4 pair-subset experiment status files into a master table, plots, and report.",
    )
    parser.add_argument(
        "--experiment-root",
        default="outputs/m4_pair_subset_experiment",
        help="Root directory for pair-subset outputs.",
    )
    parser.add_argument(
        "--pairs-csv",
        default="",
        help="Optional explicit pair manifest path. Defaults to the experiment manifest.",
    )
    parser.add_argument(
        "--baseline-summary-csv",
        default=str(DEFAULT_FULL_BASELINE_SUMMARY),
        help="Existing standardized baseline summary used for gain/loss plots.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of top-performing pairs to highlight in the top-pairs plot.",
    )
    return parser.parse_args()


def float_or_nan(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def load_status_rows(experiment_root: Path, pairs_csv: Path, baseline_csv: Path) -> tuple[list[dict[str, object]], dict[str, float | str] | None]:
    jobs = load_pair_jobs(pairs_csv)
    baseline_row = load_baseline_summary_row(baseline_csv)

    rows: list[dict[str, object]] = []
    for job in jobs:
        status_path = pair_status_path(experiment_root, job)
        if status_path.exists():
            status = read_json(status_path)
        else:
            status = {
                "subset": {"status": "pending"},
                "training": {"status": "pending"},
                "evaluations": {
                    "option_a_full_test": {"status": "pending"},
                    "option_b_pair_test": {"status": "pending"},
                },
            }

        subset = status.get("subset", {})
        training = status.get("training", {})
        evaluations = status.get("evaluations", {})
        option_a = evaluations.get("option_a_full_test", {})
        option_b = evaluations.get("option_b_pair_test", {})
        option_a_metrics = option_a.get("metrics", {})
        option_b_metrics = option_b.get("metrics", {})
        split_counts = subset.get("split_counts", {})

        row: dict[str, object] = {
            "pair_id": job.pair_id,
            "viewpoint_1": job.viewpoint_1,
            "viewpoint_2": job.viewpoint_2,
            "number_of_train_images": split_counts.get("train_images", ""),
            "number_of_val_images": split_counts.get("val_images", ""),
            "number_of_test_images": option_a_metrics.get("num_test_images", split_counts.get("test_full_images", "")),
            "precision": float_or_nan(option_a_metrics.get("precision")),
            "recall": float_or_nan(option_a_metrics.get("recall")),
            "F1": float_or_nan(option_a_metrics.get("f1")),
            "mAP50": float_or_nan(option_a_metrics.get("map50")),
            "mAP50-95": float_or_nan(option_a_metrics.get("map50_95")),
            "training_status": training.get("status", "pending"),
            "model_path": training.get("model_path", ""),
            "subset_status": subset.get("status", "pending"),
            "option_a_status": option_a.get("status", "pending"),
            "option_b_status": option_b.get("status", "pending"),
            "option_b_number_of_test_images": option_b_metrics.get("num_test_images", split_counts.get("test_pair_images", "")),
            "option_b_precision": float_or_nan(option_b_metrics.get("precision")),
            "option_b_recall": float_or_nan(option_b_metrics.get("recall")),
            "option_b_F1": float_or_nan(option_b_metrics.get("f1")),
            "option_b_mAP50": float_or_nan(option_b_metrics.get("map50")),
            "option_b_mAP50-95": float_or_nan(option_b_metrics.get("map50_95")),
            "pilot_rank": "" if job.pilot_rank is None else job.pilot_rank,
            "pilot_name": job.pilot_name,
            "pilot_note": job.pilot_note,
            "recommended_protocol": DEFAULT_PROTOCOL_RECOMMENDATION,
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
    return [row for row in rows if row.get("option_a_status") == "completed" and not math.isnan(float_or_nan(row.get("mAP50-95")))]


def save_master_tables(rows: list[dict[str, object]], experiment_root: Path) -> None:
    output_dir = experiment_root / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    base_fields = [
        "pair_id",
        "viewpoint_1",
        "viewpoint_2",
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
        "option_a_status",
        "option_b_status",
        "option_b_number_of_test_images",
        "option_b_precision",
        "option_b_recall",
        "option_b_F1",
        "option_b_mAP50",
        "option_b_mAP50-95",
        "delta_mAP50-95_vs_full_M4",
        "delta_mAP50_vs_full_M4",
        "pilot_rank",
        "pilot_name",
        "pilot_note",
        "recommended_protocol",
    ]
    write_csv_rows(output_dir / "master_results.csv", base_fields, rows)
    write_csv_rows(output_dir / "master_results_option_a.csv", base_fields, rows)

    option_b_fields = [
        "pair_id",
        "viewpoint_1",
        "viewpoint_2",
        "number_of_train_images",
        "number_of_val_images",
        "option_b_number_of_test_images",
        "option_b_precision",
        "option_b_recall",
        "option_b_F1",
        "option_b_mAP50",
        "option_b_mAP50-95",
        "training_status",
        "model_path",
        "option_b_status",
        "pilot_rank",
        "pilot_name",
        "pilot_note",
    ]
    write_csv_rows(output_dir / "master_results_option_b.csv", option_b_fields, rows)


def plot_top_pairs(rows: list[dict[str, object]], output_path: Path, top_k: int) -> None:
    completed = sorted(completed_rows(rows), key=lambda row: float(row["mAP50-95"]), reverse=True)[:top_k]
    fig, ax = plt.subplots(figsize=(14, max(6, len(completed) * 0.35)))
    if not completed:
        ax.text(0.5, 0.5, "No completed pair evaluations yet.", ha="center", va="center")
        ax.axis("off")
    else:
        labels = [
            f"{row['pair_id']}: {short_viewpoint_label(str(row['viewpoint_1']))} + {short_viewpoint_label(str(row['viewpoint_2']))}"
            for row in completed
        ]
        values = [float(row["mAP50-95"]) for row in completed]
        ax.barh(labels[::-1], values[::-1], color="#1f77b4")
        ax.set_xlabel("mAP50-95 on full fixed M4 test")
        ax.set_title("Top-performing viewpoint pairs")
        ax.grid(axis="x", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_distribution(rows: list[dict[str, object]], output_path: Path) -> None:
    completed = completed_rows(rows)
    fig, ax = plt.subplots(figsize=(10, 6))
    if not completed:
        ax.text(0.5, 0.5, "No completed pair evaluations yet.", ha="center", va="center")
        ax.axis("off")
    else:
        values = [float(row["mAP50-95"]) for row in completed]
        ax.hist(values, bins=25, color="#ff7f0e", edgecolor="white", alpha=0.9)
        ax.set_xlabel("mAP50-95 on full fixed M4 test")
        ax.set_ylabel("Number of pairs")
        ax.set_title("Distribution of pair performance")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_heatmap(rows: list[dict[str, object]], output_path: Path) -> None:
    viewpoints = sorted(
        set(str(row["viewpoint_1"]) for row in rows) | set(str(row["viewpoint_2"]) for row in rows),
        key=viewpoint_sort_key,
    )
    index_by_view = {view: idx for idx, view in enumerate(viewpoints)}
    grid = np.full((len(viewpoints), len(viewpoints)), np.nan, dtype=float)

    for row in completed_rows(rows):
        i = index_by_view[str(row["viewpoint_1"])]
        j = index_by_view[str(row["viewpoint_2"])]
        grid[i, j] = float(row["mAP50-95"])
        grid[j, i] = float(row["mAP50-95"])

    fig, ax = plt.subplots(figsize=(18, 16))
    if np.isnan(grid).all():
        ax.text(0.5, 0.5, "No completed pair evaluations yet.", ha="center", va="center")
        ax.axis("off")
    else:
        image = ax.imshow(grid, cmap="viridis", interpolation="nearest")
        labels = [short_viewpoint_label(view) for view in viewpoints]
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_yticklabels(labels, fontsize=6)
        ax.set_title("Pair performance heatmap (mAP50-95, full fixed M4 test)")
        cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("mAP50-95")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_gain_loss(rows: list[dict[str, object]], baseline_row: dict[str, float | str] | None, output_path: Path) -> None:
    completed = completed_rows(rows)
    fig, ax = plt.subplots(figsize=(10, 6))
    if not completed or baseline_row is None:
        ax.text(0.5, 0.5, "Baseline or completed pair results unavailable.", ha="center", va="center")
        ax.axis("off")
    else:
        values = [float(row["delta_mAP50-95_vs_full_M4"]) for row in completed]
        ax.hist(values, bins=25, color="#2ca02c", edgecolor="white", alpha=0.9)
        ax.axvline(0.0, color="#d62728", linestyle="--", linewidth=1.5)
        ax.set_xlabel("Delta mAP50-95 vs full M4 YOLOv8l baseline")
        ax.set_ylabel("Number of pairs")
        ax.set_title("Gain/loss relative to the full M4 model")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_report(rows: list[dict[str, object]], baseline_row: dict[str, float | str] | None, output_path: Path) -> None:
    total_pairs = len(rows)
    completed = completed_rows(rows)
    pilot_rows = sorted(
        [row for row in rows if str(row.get("pilot_rank", "")).strip()],
        key=lambda row: int(row["pilot_rank"]),
    )

    lines = [
        "# M4 Pair-Subset Experiment Report",
        "",
        "## What Was Trained",
        "",
        "- One normal single-image YOLOv8l detector per viewpoint pair.",
        "- Each pair model is trained only on M4 `train` images whose filenames match the selected two viewpoints.",
        "- Validation during training uses the matching pair-filtered `val` split.",
        "- Labels are preserved exactly by reusing the original M4 label files through YOLO list files instead of copying annotations.",
        "",
        "## Scientific Question",
        "",
        "- This experiment asks how much detector generalization can be learned from training on only two viewpoints.",
        "- It does not measure multi-view inference or fusion; the model architecture remains unchanged and each image is still evaluated independently.",
        "",
        "## Evaluation Protocol",
        "",
        "- `Option A` (recommended): evaluate every pair-trained model on the full fixed M4 test split across all 72 viewpoints.",
        "- `Option B`: evaluate each pair-trained model only on the same two viewpoints in the fixed M4 test split.",
        "- Recommendation: use `Option A` as the headline comparison because it measures generalization from a restricted training subset to the full viewpoint space.",
        "- `Option B` is still useful as a diagnostic for in-subset fit but should not be treated as the primary scientific result.",
        "",
        "## Current Sweep Status",
        "",
        f"- Pair definitions: {total_pairs}",
        f"- Completed Option A evaluations: {len(completed)}",
        f"- Recommended default metric source: {DEFAULT_PROTOCOL_RECOMMENDATION}",
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
        lines.extend(
            [
                "",
                "## Best Completed Pair So Far",
                "",
                f"- Pair: `{best_row['pair_id']}`",
                f"- Viewpoints: `{best_row['viewpoint_1']}` + `{best_row['viewpoint_2']}`",
                f"- Option A `mAP50-95`: {float(best_row['mAP50-95']):.4f}",
                f"- Option A `mAP50`: {float(best_row['mAP50']):.4f}",
                f"- Option A `F1`: {float(best_row['F1']):.4f}",
            ]
        )

    if pilot_rows:
        lines.extend(["", "## Pilot Pairs", ""])
        for row in pilot_rows:
            lines.append(
                f"- Pilot {int(row['pilot_rank'])}: `{row['pair_id']}` = `{row['viewpoint_1']}` + `{row['viewpoint_2']}` ({row['pilot_name']})"
            )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    experiment_root = Path(args.experiment_root).resolve()
    ensure_experiment_root(experiment_root)
    pairs_csv = Path(args.pairs_csv).resolve() if args.pairs_csv else pair_manifest_path(experiment_root)
    baseline_csv = Path(args.baseline_summary_csv).resolve()

    rows, baseline_row = load_status_rows(experiment_root, pairs_csv, baseline_csv)
    save_master_tables(rows, experiment_root)

    plots_dir = experiment_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_top_pairs(rows, plots_dir / "top_performing_pairs.png", top_k=args.top_k)
    plot_distribution(rows, plots_dir / "pair_performance_distribution.png")
    plot_heatmap(rows, plots_dir / "pair_performance_heatmap.png")
    plot_gain_loss(rows, baseline_row, plots_dir / "gain_loss_vs_full_m4.png")

    report_path = experiment_root / "reports" / "pair_subset_experiment_report.md"
    write_report(rows, baseline_row, report_path)
    print(f"Wrote master tables, plots, and report under: {experiment_root}")


if __name__ == "__main__":
    main()
