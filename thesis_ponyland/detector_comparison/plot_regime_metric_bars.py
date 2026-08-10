from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from comparison_config import DETECTOR_ORDER, REGIME_ORDER


METRICS = [
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("f1", "F1"),
    ("map50", "mAP50"),
    ("map50_95", "mAP50-95"),
    ("ap75", "AP75"),
    ("matched_mean_iou", "Mean IoU"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot regime-specific grouped metric bars from the standardized comparison CSV.",
    )
    parser.add_argument("--summary-csv", required=True, help="Path to standardized comparison summary CSV.")
    parser.add_argument("--output", required=True, help="PNG path for the regime-by-regime comparison figure.")
    parser.add_argument("--title", default="3-model comparison by regime and metric", help="Figure title.")
    return parser.parse_args()


def load_summary_rows(summary_csv: Path) -> list[dict[str, str]]:
    with summary_csv.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def plot_regime_metric_bars(rows: list[dict[str, str]], output_path: Path, title: str) -> None:
    lookup = {(row["regime"], row["detector"]): row for row in rows}
    colors = {
        "YOLOv8n": "#4C78A8",
        "YOLOv8l": "#F58518",
        "Faster R-CNN": "#54A24B",
    }

    fig, axes = plt.subplots(len(REGIME_ORDER), 1, figsize=(18, 20), sharey=True)
    width = 0.24
    x = np.arange(len(METRICS))

    for axis, regime in zip(axes, REGIME_ORDER):
        for detector_idx, detector in enumerate(DETECTOR_ORDER):
            row = lookup.get((regime, detector))
            if row is None:
                continue
            values = [float(row[metric]) for metric, _ in METRICS]
            offset = (detector_idx - 1) * width
            axis.bar(
                x + offset,
                values,
                width=width,
                color=colors.get(detector),
                label=detector if regime == REGIME_ORDER[0] else None,
            )

        axis.set_title(f"{regime}", loc="left", fontsize=13, fontweight="bold")
        axis.set_xticks(x)
        axis.set_xticklabels([label for _, label in METRICS], rotation=20, ha="right")
        axis.set_ylim(0, 1)
        axis.set_ylabel("Score")
        axis.grid(axis="y", linestyle="--", alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle(title, fontsize=18)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    summary_csv = Path(args.summary_csv).resolve()
    output_path = Path(args.output).resolve()
    rows = load_summary_rows(summary_csv)
    plot_regime_metric_bars(rows, output_path, args.title)
    print(f"Saved regime-by-regime comparison figure to: {output_path}")


if __name__ == "__main__":
    main()
