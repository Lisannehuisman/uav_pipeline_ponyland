from __future__ import annotations

import argparse
from pathlib import Path

from pair_experiment_lib import (
    collect_all_viewpoints,
    ensure_experiment_root,
    enumerate_pair_jobs,
    pair_manifest_path,
    viewpoint_counts_by_split,
    write_csv_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enumerate all M4 viewpoint pairs and write a reproducible pair manifest.",
    )
    parser.add_argument(
        "--base-data-yaml",
        default=r"C:\DATA\airsim\thesis\captures\S0_20251219_164144\dataset\M4_fixed.yaml",
        help="Base M4 dataset YAML used to discover viewpoints and split counts.",
    )
    parser.add_argument(
        "--experiment-root",
        default="outputs/m4_pair_subset_experiment",
        help="Root directory for manifests and generated experiment assets.",
    )
    parser.add_argument(
        "--pilot-count",
        type=int,
        default=5,
        help="Number of pilot pairs to flag in the manifest.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_data_yaml = Path(args.base_data_yaml).resolve()
    experiment_root = Path(args.experiment_root).resolve()
    ensure_experiment_root(experiment_root)

    viewpoints = collect_all_viewpoints(base_data_yaml)
    split_counts = viewpoint_counts_by_split(base_data_yaml)
    pair_jobs = enumerate_pair_jobs(viewpoints, pilot_count=args.pilot_count)

    rows: list[dict[str, object]] = []
    for job in pair_jobs:
        train_images = split_counts["train"][job.viewpoint_1] + split_counts["train"][job.viewpoint_2]
        val_images = split_counts["val"][job.viewpoint_1] + split_counts["val"][job.viewpoint_2]
        test_pair_images = split_counts["test"][job.viewpoint_1] + split_counts["test"][job.viewpoint_2]
        rows.append(
            {
                "pair_index": job.pair_index,
                "pair_id": job.pair_id,
                "viewpoint_1": job.viewpoint_1,
                "viewpoint_2": job.viewpoint_2,
                "expected_train_images": train_images,
                "expected_val_images": val_images,
                "expected_test_pair_images": test_pair_images,
                "pilot_rank": "" if job.pilot_rank is None else job.pilot_rank,
                "pilot_name": job.pilot_name,
                "pilot_note": job.pilot_note,
            }
        )

    write_csv_rows(
        pair_manifest_path(experiment_root),
        fieldnames=[
            "pair_index",
            "pair_id",
            "viewpoint_1",
            "viewpoint_2",
            "expected_train_images",
            "expected_val_images",
            "expected_test_pair_images",
            "pilot_rank",
            "pilot_name",
            "pilot_note",
        ],
        rows=rows,
    )

    viewpoint_rows = []
    for viewpoint in viewpoints:
        viewpoint_rows.append(
            {
                "viewpoint": viewpoint,
                "train_images": split_counts["train"][viewpoint],
                "val_images": split_counts["val"][viewpoint],
                "test_images": split_counts["test"][viewpoint],
            }
        )

    write_csv_rows(
        experiment_root / "manifests" / "viewpoint_inventory.csv",
        fieldnames=["viewpoint", "train_images", "val_images", "test_images"],
        rows=viewpoint_rows,
    )
    print(f"Wrote {len(pair_jobs)} pair definitions to: {pair_manifest_path(experiment_root)}")


if __name__ == "__main__":
    main()

