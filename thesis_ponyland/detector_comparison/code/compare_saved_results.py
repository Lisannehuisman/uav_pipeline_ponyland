from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from comparison_config import DEFAULT_OUTPUT_DIR, DETECTOR_ORDER, MODEL_RUNS, REGIME_ORDER


def safe_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def resolve_frcnn_run_dir(run_dir: Path) -> Path:
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        return run_dir

    nested_metrics = list(run_dir.glob("*/metrics.json"))
    if nested_metrics:
        return nested_metrics[0].parent
    return run_dir


def load_yolo_saved_metrics(run_dir: Path) -> dict[str, float]:
    results_csv = run_dir / "results.csv"
    with results_csv.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    final_row = rows[-1]

    precision = safe_float(final_row.get("metrics/precision(B)"))
    recall = safe_float(final_row.get("metrics/recall(B)"))
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall > 0 else float("nan")

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "map50": safe_float(final_row.get("metrics/mAP50(B)")),
        "map50_95": safe_float(final_row.get("metrics/mAP50-95(B)")),
        "ap75": float("nan"),
    }


def load_frcnn_saved_metrics(run_dir: Path) -> dict[str, float]:
    run_dir = resolve_frcnn_run_dir(run_dir)
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return {
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "map50": float("nan"),
            "map50_95": float("nan"),
            "ap75": float("nan"),
        }
    eval_rows = []
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "bbox/AP" in row:
                eval_rows.append(row)
    final_row = eval_rows[-1]

    return {
        "precision": float("nan"),
        "recall": float("nan"),
        "f1": float("nan"),
        "map50": safe_float(final_row.get("bbox/AP50")) / 100.0,
        "map50_95": safe_float(final_row.get("bbox/AP")) / 100.0,
        "ap75": safe_float(final_row.get("bbox/AP75")) / 100.0,
    }


def plot_quick_summary(rows: list[dict[str, float | str]], output_path: Path) -> None:
    metrics = ["map50_95", "map50", "ap75"]
    titles = ["mAP50-95", "mAP50", "AP75"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

    x = np.arange(len(REGIME_ORDER))
    width = 0.24

    for ax, metric, title in zip(axes, metrics, titles):
        for idx, detector in enumerate(DETECTOR_ORDER):
            values = [
                next(
                    float(row[metric])
                    for row in rows
                    if row["detector"] == detector and row["regime"] == regime
                )
                for regime in REGIME_ORDER
            ]
            ax.bar(x + (idx - 1) * width, values, width=width, label=detector)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(REGIME_ORDER)
        ax.set_ylim(0, 1)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    axes[0].set_ylabel("Score")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.suptitle("Quick saved-results comparison", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    output_dir = DEFAULT_OUTPUT_DIR / "quick_saved_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | str]] = []
    for detector, regime_map in MODEL_RUNS.items():
        for regime in REGIME_ORDER:
            run_dir = Path(regime_map[regime])
            if detector.startswith("YOLO"):
                metrics = load_yolo_saved_metrics(run_dir)
            else:
                metrics = load_frcnn_saved_metrics(run_dir)
            rows.append({"detector": detector, "regime": regime, **metrics})

    summary_csv = output_dir / "quick_saved_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["detector", "regime", "precision", "recall", "f1", "map50", "map50_95", "ap75"],
        )
        writer.writeheader()
        writer.writerows(rows)

    plot_quick_summary(rows, output_dir / "quick_saved_summary.png")
    print(f"Saved quick comparison to: {output_dir}")


if __name__ == "__main__":
    main()
