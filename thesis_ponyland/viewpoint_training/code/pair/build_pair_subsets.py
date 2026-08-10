from __future__ import annotations

import argparse
from pathlib import Path

from pair_experiment_lib import (
    build_pair_subset,
    ensure_experiment_root,
    load_pair_jobs,
    pair_manifest_path,
    validate_pair_subset,
    write_csv_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build YOLO list-file subsets for one or more M4 viewpoint pairs.",
    )
    parser.add_argument(
        "--base-data-yaml",
        default=r"C:\DATA\airsim\thesis\captures\S0_20251219_164144\dataset\M4_fixed.yaml",
        help="Base M4 dataset YAML used to resolve train/val/test images.",
    )
    parser.add_argument(
        "--experiment-root",
        default="outputs/m4_pair_subset_experiment",
        help="Root directory for the pair-subset experiment.",
    )
    parser.add_argument(
        "--pairs-csv",
        default="",
        help="Optional explicit pair manifest path. Defaults to the experiment manifest.",
    )
    parser.add_argument(
        "--pair-ids",
        nargs="*",
        default=[],
        help="Optional list of pair ids to build. Defaults to every pair in the manifest.",
    )
    parser.add_argument(
        "--pilot-only",
        action="store_true",
        help="Build only the pilot-marked pairs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional maximum number of pairs to build after filtering.",
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
    ensure_experiment_root(experiment_root)
    base_data_yaml = Path(args.base_data_yaml).resolve()
    pairs_csv = Path(args.pairs_csv).resolve() if args.pairs_csv else pair_manifest_path(experiment_root)
    jobs = load_pair_jobs(pairs_csv)

    if args.pair_ids:
        allowed_ids = set(args.pair_ids)
        jobs = [job for job in jobs if job.pair_id in allowed_ids]
    if args.pilot_only:
        jobs = [job for job in jobs if job.pilot_rank is not None]
    if args.limit > 0:
        jobs = jobs[: args.limit]

    rows: list[dict[str, object]] = []
    for job in jobs:
        if args.validate_only:
            metadata_path = experiment_root / "pairs" / job.slug / "subset_metadata.json"
            metadata = validate_pair_subset(metadata_path)
        else:
            metadata = build_pair_subset(base_data_yaml, experiment_root, job, force=args.force)
        split_counts = metadata["split_counts"]
        rows.append(
            {
                "pair_id": job.pair_id,
                "viewpoint_1": job.viewpoint_1,
                "viewpoint_2": job.viewpoint_2,
                "train_images": split_counts["train_images"],
                "val_images": split_counts["val_images"],
                "test_pair_images": split_counts["test_pair_images"],
                "test_full_images": split_counts["test_full_images"],
                "data_yaml": metadata["subset_files"]["data_yaml"],
                "pilot_rank": "" if job.pilot_rank is None else job.pilot_rank,
            }
        )

    write_csv_rows(
        experiment_root / "manifests" / "subset_build_summary.csv",
        fieldnames=[
            "pair_id",
            "viewpoint_1",
            "viewpoint_2",
            "train_images",
            "val_images",
            "test_pair_images",
            "test_full_images",
            "data_yaml",
            "pilot_rank",
        ],
        rows=rows,
    )
    print(f"Processed {len(rows)} pair subsets. Summary: {experiment_root / 'manifests' / 'subset_build_summary.csv'}")


if __name__ == "__main__":
    main()
