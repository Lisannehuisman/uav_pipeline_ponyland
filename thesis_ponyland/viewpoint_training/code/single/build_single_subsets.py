from __future__ import annotations

import argparse
from pathlib import Path

from single_experiment_lib import (
    build_single_subset,
    ensure_single_experiment_root,
    load_single_jobs,
    single_manifest_path,
    validate_single_subset,
    write_csv_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build YOLO list-file subsets for one or more M4 training viewpoints.",
    )
    parser.add_argument(
        "--base-data-yaml",
        default=r"C:\DATA\airsim\thesis\captures\S0_20251219_164144\dataset\M4_fixed.yaml",
        help="Base M4 dataset YAML used to resolve train/val/test images.",
    )
    parser.add_argument(
        "--experiment-root",
        default="outputs/m4_single_subset_experiment",
        help="Root directory for the single-viewpoint experiment.",
    )
    parser.add_argument(
        "--singles-csv",
        default="",
        help="Optional explicit single-viewpoint manifest path. Defaults to the experiment manifest.",
    )
    parser.add_argument(
        "--single-ids",
        nargs="*",
        default=[],
        help="Optional list of single ids to build. Defaults to every viewpoint in the manifest.",
    )
    parser.add_argument(
        "--pilot-only",
        action="store_true",
        help="Build only the pilot-marked single viewpoints.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional maximum number of viewpoints to build after filtering.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild subsets even if metadata already exists.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate previously built subsets.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_root = Path(args.experiment_root).resolve()
    ensure_single_experiment_root(experiment_root)
    base_data_yaml = Path(args.base_data_yaml).resolve()
    singles_csv = Path(args.singles_csv).resolve() if args.singles_csv else single_manifest_path(experiment_root)
    jobs = load_single_jobs(singles_csv)

    if args.single_ids:
        allowed_ids = set(args.single_ids)
        jobs = [job for job in jobs if job.single_id in allowed_ids]
    if args.pilot_only:
        jobs = [job for job in jobs if job.pilot_rank is not None]
    if args.limit > 0:
        jobs = jobs[: args.limit]

    rows: list[dict[str, object]] = []
    for job in jobs:
        if args.validate_only:
            metadata_path = experiment_root / "singles" / job.slug / "subset_metadata.json"
            metadata = validate_single_subset(metadata_path)
        else:
            metadata = build_single_subset(base_data_yaml, experiment_root, job, force=args.force)
        split_counts = metadata["split_counts"]
        rows.append(
            {
                "single_id": job.single_id,
                "viewpoint": job.viewpoint,
                "train_images": split_counts["train_images"],
                "val_images": split_counts["val_images"],
                "test_view_images": split_counts["test_view_images"],
                "test_full_images": split_counts["test_full_images"],
                "data_yaml": metadata["subset_files"]["data_yaml"],
                "pilot_rank": "" if job.pilot_rank is None else job.pilot_rank,
            }
        )

    write_csv_rows(
        experiment_root / "manifests" / "subset_build_summary.csv",
        fieldnames=[
            "single_id",
            "viewpoint",
            "train_images",
            "val_images",
            "test_view_images",
            "test_full_images",
            "data_yaml",
            "pilot_rank",
        ],
        rows=rows,
    )
    print(
        f"Processed {len(rows)} single-viewpoint subsets. "
        f"Summary: {experiment_root / 'manifests' / 'subset_build_summary.csv'}"
    )


if __name__ == "__main__":
    main()
