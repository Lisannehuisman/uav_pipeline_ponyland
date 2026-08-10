from __future__ import annotations

import argparse
from pathlib import Path

from matched_control_lib import (
    DEFAULT_CONTROL_EXPERIMENT_ROOT,
    control_manifest_path,
    ensure_control_experiment_root,
    write_csv_rows,
)


DEFAULT_SINGLE_RESULTS = Path("viewpoint_data_separated") / "72_trained_models" / "reports" / "master_results.csv"
DEFAULT_PAIR_RESULTS = Path("m4_pair_results") / "data" / "current_snapshot" / "reports" / "master_results.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a manifest of equal-image-count full-M4 control jobs for the current best single and pair models.",
    )
    parser.add_argument("--experiment-root", default=str(DEFAULT_CONTROL_EXPERIMENT_ROOT))
    parser.add_argument("--single-results-csv", default=str(DEFAULT_SINGLE_RESULTS))
    parser.add_argument("--pair-results-csv", default=str(DEFAULT_PAIR_RESULTS))
    parser.add_argument("--single-id", default="", help="Optional explicit single-view id to match instead of the best row.")
    parser.add_argument("--pair-id", default="", help="Optional explicit pair id to match instead of the best row.")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0],
        help="One or more sampling seeds for the matched M4 controls.",
    )
    parser.add_argument("--skip-single", action="store_true", help="Do not emit a matched control for the single-view source.")
    parser.add_argument("--skip-pair", action="store_true", help="Do not emit a matched control for the pair-view source.")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    import csv

    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def pick_single_row(rows: list[dict[str, str]], single_id: str) -> dict[str, str]:
    completed = [row for row in rows if row.get("evaluation_status") == "completed"]
    if single_id:
        for row in completed:
            if row.get("single_id") == single_id:
                return row
        raise KeyError(f"Single-view id '{single_id}' was not found among completed rows.")
    return max(completed, key=lambda row: float(row["mAP50-95"]))


def pick_pair_row(rows: list[dict[str, str]], pair_id: str) -> dict[str, str]:
    completed = [row for row in rows if row.get("option_a_status") == "completed"]
    if pair_id:
        for row in completed:
            if row.get("pair_id") == pair_id:
                return row
        raise KeyError(f"Pair id '{pair_id}' was not found among completed rows.")
    return max(completed, key=lambda row: float(row["mAP50-95"]))


def main() -> None:
    args = parse_args()
    experiment_root = Path(args.experiment_root).resolve()
    ensure_control_experiment_root(experiment_root)

    single_rows = read_rows(Path(args.single_results_csv).resolve())
    pair_rows = read_rows(Path(args.pair_results_csv).resolve())

    manifest_rows: list[dict[str, object]] = []
    next_index = 1

    if not args.skip_single:
        single_row = pick_single_row(single_rows, args.single_id)
        for seed in args.seeds:
            manifest_rows.append(
                {
                    "control_index": next_index,
                    "control_id": f"mc_single_best_s{seed:02d}",
                    "label": f"M4 matched to best single (seed {seed})",
                    "source_group": "single",
                    "source_id": single_row["single_id"],
                    "source_label": single_row["viewpoint"],
                    "train_count": int(single_row["number_of_train_images"]),
                    "val_count": int(single_row["number_of_val_images"]),
                    "seed": seed,
                    "sampling_strategy": "stratified_viewpoint",
                    "reference_mAP50-95": float(single_row["mAP50-95"]),
                    "reference_mAP50": float(single_row["mAP50"]),
                    "reference_F1": float(single_row["F1"]),
                }
            )
            next_index += 1

    if not args.skip_pair:
        pair_row = pick_pair_row(pair_rows, args.pair_id)
        for seed in args.seeds:
            manifest_rows.append(
                {
                    "control_index": next_index,
                    "control_id": f"mc_pair_best_s{seed:02d}",
                    "label": f"M4 matched to best pair (seed {seed})",
                    "source_group": "pair",
                    "source_id": pair_row["pair_id"],
                    "source_label": f"{pair_row['viewpoint_1']} + {pair_row['viewpoint_2']}",
                    "train_count": int(pair_row["number_of_train_images"]),
                    "val_count": int(pair_row["number_of_val_images"]),
                    "seed": seed,
                    "sampling_strategy": "stratified_viewpoint",
                    "reference_mAP50-95": float(pair_row["mAP50-95"]),
                    "reference_mAP50": float(pair_row["mAP50"]),
                    "reference_F1": float(pair_row["F1"]),
                }
            )
            next_index += 1

    if not manifest_rows:
        raise SystemExit("No matched-control jobs were requested.")

    manifest_path = control_manifest_path(experiment_root)
    write_csv_rows(
        manifest_path,
        [
            "control_index",
            "control_id",
            "label",
            "source_group",
            "source_id",
            "source_label",
            "train_count",
            "val_count",
            "seed",
            "sampling_strategy",
            "reference_mAP50-95",
            "reference_mAP50",
            "reference_F1",
        ],
        manifest_rows,
    )

    print(f"Wrote matched-control manifest: {manifest_path}")
    for row in manifest_rows:
        print(
            f"{row['control_id']}: {row['source_group']} {row['source_id']} -> "
            f"{row['train_count']} train / {row['val_count']} val images (seed={row['seed']})"
        )


if __name__ == "__main__":
    main()
