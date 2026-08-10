from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from comparison_config import DETECTOR_ORDER, REGIME_ORDER


METRIC_PRIORITY = [
    ("map50_95", "mAP50-95"),
    ("f1", "F1"),
    ("matched_mean_iou", "Mean IoU"),
    ("ap75", "AP75"),
    ("map50", "mAP50"),
    ("precision", "Precision"),
    ("recall", "Recall"),
]

SHORT_NAMES = {
    "YOLOv8n": "n",
    "YOLOv8l": "l",
    "Faster R-CNN": "fr",
}

WINNER_COLORS = {
    "YOLOv8n": "#DCEAF7",
    "YOLOv8l": "#FCE6D3",
    "Faster R-CNN": "#DDEFD9",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a large regime-by-metric summary table from the standardized comparison CSV.",
    )
    parser.add_argument("--summary-csv", required=True, help="Path to the standardized summary CSV.")
    parser.add_argument("--output-csv", required=True, help="Path to save the wide summary CSV.")
    parser.add_argument("--output-png", required=True, help="Path to save the table PNG.")
    parser.add_argument(
        "--title",
        default="Validation comparison table across regimes and metrics",
        help="Title for the table figure.",
    )
    return parser.parse_args()


def load_rows(summary_csv: Path) -> list[dict[str, str]]:
    with summary_csv.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def format_cell(metric: str, regime_rows: dict[str, dict[str, str]]) -> tuple[str, str]:
    values = {detector: float(regime_rows[detector][metric]) for detector in DETECTOR_ORDER}
    winner = max(values, key=values.get)
    parts = [f"{SHORT_NAMES[detector]} {values[detector]:.3f}" for detector in DETECTOR_ORDER]
    text = f"Best: {winner} {values[winner]:.3f}\n" + " | ".join(parts)
    return text, WINNER_COLORS[winner]


def average_rows(rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    averages: dict[str, dict[str, float]] = {}
    for detector in DETECTOR_ORDER:
        detector_rows = [row for row in rows if row["detector"] == detector]
        averages[detector] = {
            metric: sum(float(row[metric]) for row in detector_rows) / len(detector_rows) for metric, _ in METRIC_PRIORITY
        }
    return averages


def save_wide_csv(rows: list[dict[str, str]], output_csv: Path) -> None:
    lookup = {(row["regime"], row["detector"]): row for row in rows}
    averages = average_rows(rows)

    fieldnames = ["metric", "metric_label", *REGIME_ORDER, "overall_avg"]
    table_rows: list[dict[str, str]] = []

    for metric, metric_label in METRIC_PRIORITY:
        row_out = {"metric": metric, "metric_label": metric_label}
        for regime in REGIME_ORDER:
            regime_rows = {detector: lookup[(regime, detector)] for detector in DETECTOR_ORDER}
            cell_text, _ = format_cell(metric, regime_rows)
            row_out[regime] = cell_text.replace("\n", " ")

        avg_winner = max(DETECTOR_ORDER, key=lambda detector: averages[detector][metric])
        avg_parts = [f"{SHORT_NAMES[detector]} {averages[detector][metric]:.3f}" for detector in DETECTOR_ORDER]
        row_out["overall_avg"] = (
            f"Best: {avg_winner} {averages[avg_winner][metric]:.3f} " + " | ".join(avg_parts)
        )
        table_rows.append(row_out)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(table_rows)


def create_table_figure(rows: list[dict[str, str]], output_png: Path, title: str) -> None:
    lookup = {(row["regime"], row["detector"]): row for row in rows}
    averages = average_rows(rows)

    headers = ["Metric", *REGIME_ORDER, "Overall Avg"]
    cell_text: list[list[str]] = []
    cell_colors: list[list[str]] = []

    for metric, metric_label in METRIC_PRIORITY:
        row_text = [metric_label]
        row_colors = ["#F3F4F6"]

        for regime in REGIME_ORDER:
            regime_rows = {detector: lookup[(regime, detector)] for detector in DETECTOR_ORDER}
            text, color = format_cell(metric, regime_rows)
            row_text.append(text)
            row_colors.append(color)

        avg_winner = max(DETECTOR_ORDER, key=lambda detector: averages[detector][metric])
        avg_parts = [f"{SHORT_NAMES[detector]} {averages[detector][metric]:.3f}" for detector in DETECTOR_ORDER]
        row_text.append(f"Best: {avg_winner} {averages[avg_winner][metric]:.3f}\n" + " | ".join(avg_parts))
        row_colors.append(WINNER_COLORS[avg_winner])

        cell_text.append(row_text)
        cell_colors.append(row_colors)

    fig, ax = plt.subplots(figsize=(26, 10))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        colLabels=headers,
        cellColours=cell_colors,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.8)

    for col_idx in range(len(headers)):
        header_cell = table[(0, col_idx)]
        header_cell.set_facecolor("#D9E2F3")
        header_cell.set_text_props(weight="bold")

    for row_idx in range(1, len(METRIC_PRIORITY) + 1):
        metric_cell = table[(row_idx, 0)]
        metric_cell.set_text_props(weight="bold")

    fig.suptitle(title, fontsize=18, y=0.97)
    fig.text(
        0.01,
        0.02,
        "Cell format: winning model shown first, followed by n/l/fr scores. "
        "Metric order: mAP50-95, F1, Mean IoU, AP75, mAP50, Precision, Recall.",
        fontsize=10,
    )
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(output_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    summary_csv = Path(args.summary_csv).resolve()
    output_csv = Path(args.output_csv).resolve()
    output_png = Path(args.output_png).resolve()

    rows = load_rows(summary_csv)
    save_wide_csv(rows, output_csv)
    create_table_figure(rows, output_png, args.title)
    print(f"Saved wide summary CSV to: {output_csv}")
    print(f"Saved table PNG to: {output_png}")


if __name__ == "__main__":
    main()
