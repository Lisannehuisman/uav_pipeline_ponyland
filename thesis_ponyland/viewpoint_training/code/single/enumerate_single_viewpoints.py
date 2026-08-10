from __future__ import annotations

import argparse
from pathlib import Path

from single_experiment_lib import (
    collect_all_viewpoints,
    ensure_single_experiment_root,
    enumerate_single_jobs,
    single_manifest_path,
    viewpoint_counts_by_split,
    write_csv_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enumerate all M4 training viewpoints and write a reproducible single-viewpoint manifest.",
    )
    parser.add_argument(
        "--base-data-yaml",
        default=r"C:\DATA\airsim\thesis\captures\S0_20251219_164144\dataset\M4_fixed.yaml",
        help="Base M4 dataset YAML used to discover viewpoints and split counts.",
    )
    parser.add_argument(
        "--experiment-root",
        default="outputs/m4_single_subset_experiment",
        help="Root directory for manifests and generated experiment assets.",
    )
    parser.add_argument(
        "--pilot-count",
        type=int,
        default=5,
        help="Number of pilot viewpoints to flag in the manifest.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_data_yaml = Path(args.base_data_yaml).resolve()
    experiment_root = Path(args.experiment_root).resolve()
    ensure_single_experiment_root(experiment_root)

    viewpoints = collect_all_viewpoints(base_data_yaml)
    split_counts = viewpoint_counts_by_split(base_data_yaml)
    single_jobs = enumerate_single_jobs(viewpoints, pilot_count=args.pilot_count)

    manifest_rows: list[dict[str, object]] = []
    for job in single_jobs:
        manifest_rows.append(
            {
                "viewpoint_index": job.viewpoint_index,
                "single_id": job.single_id,
                "viewpoint": job.viewpoint,
                "expected_train_images": split_counts["train"][job.viewpoint],
                "expected_val_images": split_counts["val"][job.viewpoint],
                "expected_test_view_images": split_counts["test"][job.viewpoint],
                "pilot_rank": "" if job.pilot_rank is None else job.pilot_rank,
                "pilot_name": job.pilot_name,
                "pilot_note": job.pilot_note,
            }
        )

    write_csv_rows(
        single_manifest_path(experiment_root),
        fieldnames=[
            "viewpoint_index",
            "single_id",
            "viewpoint",
            "expected_train_images",
            "expected_val_images",
            "expected_test_view_images",
            "pilot_rank",
            "pilot_name",
            "pilot_note",
        ],
        rows=manifest_rows,
    )

    viewpoint_rows: list[dict[str, object]] = []
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
    print(f"Wrote {len(single_jobs)} single-viewpoint definitions to: {single_manifest_path(experiment_root)}")


if __name__ == "__main__":
    main()

