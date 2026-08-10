from __future__ import annotations

import argparse
from pathlib import Path

from comparison_config import DEFAULT_OUTPUT_DIR, DETECTOR_ORDER, REGIME_DATA_YAMLS, REGIME_ORDER
from create_regime_metric_table import create_table_figure, load_rows as load_table_rows, save_wide_csv
from plot_per_class_regime_bars import (
    category_names_from_gt,
    detector_prediction_json,
    per_class_ap,
    plot_regime_grid,
    resolve_official_coco_gt,
    save_per_class_csv,
)
from plot_regime_metric_bars import load_summary_rows, plot_regime_metric_bars


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate thesis-ready summary figures and tables from a standardized evaluation run.",
    )
    parser.add_argument("--split", choices=["val", "test"], default="test", help="Shared split to visualize.")
    parser.add_argument(
        "--per-class-metrics",
        nargs="+",
        choices=["ap50_95", "ap50", "ap75"],
        default=["ap50_95"],
        help="Per-class AP metrics to export.",
    )
    parser.add_argument(
        "--title-prefix",
        default=None,
        help="Optional prefix for generated figure titles.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = (DEFAULT_OUTPUT_DIR / f"standardized_{args.split}_eval").resolve()
    summary_csv = output_dir / "standardized_test_summary.csv"
    if not summary_csv.exists():
        raise FileNotFoundError(f"Missing standardized summary CSV: {summary_csv}")

    split_label = "Validation" if args.split == "val" else "Test"
    title_prefix = args.title_prefix or split_label

    summary_rows = load_summary_rows(summary_csv)
    regime_metric_output = output_dir / "regime_metric_comparison.png"
    plot_regime_metric_bars(
        summary_rows,
        regime_metric_output,
        f"{title_prefix} comparison by regime and metric",
    )

    table_rows = load_table_rows(summary_csv)
    table_csv = output_dir / "regime_metric_table.csv"
    table_png = output_dir / "regime_metric_table.png"
    save_wide_csv(table_rows, table_csv)
    create_table_figure(
        table_rows,
        table_png,
        f"{title_prefix} comparison table across regimes and metrics",
    )

    per_class_dir = output_dir / "per_class_plots"
    per_class_dir.mkdir(parents=True, exist_ok=True)
    reference_gt = resolve_official_coco_gt(Path(REGIME_DATA_YAMLS[REGIME_ORDER[0]]), args.split)
    class_names = category_names_from_gt(reference_gt)

    for metric in args.per_class_metrics:
        rows: list[dict[str, str | float]] = []
        for regime in REGIME_ORDER:
            gt_json = resolve_official_coco_gt(Path(REGIME_DATA_YAMLS[regime]), args.split)
            for detector in DETECTOR_ORDER:
                pred_json = detector_prediction_json(detector, regime, args.split)
                scores = per_class_ap(gt_json, pred_json, metric=metric)
                for class_name in class_names:
                    rows.append(
                        {
                            "regime": regime,
                            "detector": detector,
                            "class_name": class_name,
                            "metric": metric,
                            "score": scores[class_name],
                        }
                    )

        output_csv = per_class_dir / f"per_class_{metric}_{args.split}.csv"
        output_png = per_class_dir / f"per_class_{metric}_{args.split}.png"
        save_per_class_csv(rows, output_csv)
        plot_regime_grid(rows, class_names, metric, output_png)

    print(f"Saved report assets under: {output_dir}")


if __name__ == "__main__":
    main()
