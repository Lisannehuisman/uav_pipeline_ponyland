from __future__ import annotations

import csv
import hashlib
import importlib
import json
import random
import re
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIEWPOINT_RE = re.compile(r"-(el[a-z]+-rad[a-z]+-az\d+)(?=\.[^.]+$)", re.IGNORECASE)
VIEWPOINT_PARTS_RE = re.compile(r"^(el[a-z]+)-(rad[a-z]+)-az(\d+)$", re.IGNORECASE)
ELEVATION_SORT = {"ellow": 0, "elmid": 1, "elhigh": 2}
RADIUS_SORT = {"radnear": 0, "radmid": 1, "radfar": 2}
DEFAULT_CONTROL_EXPERIMENT_ROOT = Path("outputs") / "m4_matched_control_experiment"
DEFAULT_FULL_BASELINE_SUMMARY = (
    Path("outputs") / "detector_family_comparison" / "standardized_test_eval" / "standardized_test_summary.csv"
)
PAIR_LIB_DIR = Path(__file__).resolve().parents[1] / "m4_pair_subset_experiment_code_2556models"


def _pair_lib():
    if str(PAIR_LIB_DIR) not in sys.path:
        sys.path.insert(0, str(PAIR_LIB_DIR))
    return importlib.import_module("pair_experiment_lib")


@dataclass(frozen=True)
class MatchedControlJob:
    control_index: int
    control_id: str
    label: str
    source_group: str
    source_id: str
    source_label: str
    train_count: int
    val_count: int
    seed: int
    sampling_strategy: str
    reference_map50_95: float
    reference_map50: float
    reference_f1: float

    @property
    def slug(self) -> str:
        return self.control_id


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_seed(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:16], 16)


def parse_viewpoint(view_id: str) -> tuple[str, str, int]:
    match = VIEWPOINT_PARTS_RE.match(view_id.strip().lower())
    if match is None:
        raise ValueError(f"Could not parse viewpoint id: {view_id}")
    elevation, radius, azimuth = match.groups()
    return elevation, radius, int(azimuth)


def viewpoint_sort_key(view_id: str) -> tuple[int, int, int]:
    elevation, radius, azimuth = parse_viewpoint(view_id)
    return (ELEVATION_SORT.get(elevation, 99), RADIUS_SORT.get(radius, 99), azimuth)


def read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=str(path.parent)) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.flush()
        temp_path = Path(handle.name)
    temp_path.replace(path)


def write_csv_rows(path: Path, fieldnames: Sequence[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def ensure_control_experiment_root(experiment_root: Path) -> None:
    for relative in ("manifests", "controls", "shared", "reports", "plots", "launchers", "logs"):
        (experiment_root / relative).mkdir(parents=True, exist_ok=True)


def resolve_dataset_root(data_yaml: Path, data_dict: dict) -> Path:
    configured_root = data_dict.get("path")
    if configured_root:
        root = Path(configured_root)
        if not root.is_absolute():
            root = (data_yaml.parent / root).resolve()
        return root
    return data_yaml.parent.resolve()


def normalize_names(data_dict: dict) -> list[str]:
    names = data_dict.get("names", {})
    if isinstance(names, dict):
        ordered = sorted(((int(key), value) for key, value in names.items()), key=lambda item: item[0])
        return [str(name) for _, name in ordered]
    return [str(name) for name in names]


def resolve_split_images(data_yaml: Path, split: str) -> list[Path]:
    data_dict = read_yaml(data_yaml)
    split_value = data_dict.get(split)
    if split_value is None:
        raise ValueError(f"Split '{split}' was not found in {data_yaml}.")

    dataset_root = resolve_dataset_root(data_yaml, data_dict)
    candidates = [Path(split_value)] if isinstance(split_value, str) else [Path(item) for item in split_value]
    images: list[Path] = []

    for candidate in candidates:
        resolved = candidate if candidate.is_absolute() else (dataset_root / candidate).resolve()
        if resolved.is_dir():
            for image_path in sorted(resolved.rglob("*")):
                if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    images.append(image_path.resolve())
            continue

        if resolved.is_file() and resolved.suffix.lower() == ".txt":
            with resolved.open("r", encoding="utf-8") as handle:
                for line in handle:
                    raw = line.strip()
                    if not raw:
                        continue
                    listed_path = Path(raw)
                    if not listed_path.is_absolute():
                        listed_path = (resolved.parent / listed_path).resolve()
                    if listed_path.suffix.lower() in IMAGE_EXTENSIONS:
                        images.append(listed_path.resolve())
            continue

        if resolved.is_file() and resolved.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(resolved.resolve())
            continue

        raise FileNotFoundError(f"Could not resolve image path '{candidate}' from split '{split}'.")

    deduplicated: list[Path] = []
    seen: set[str] = set()
    for image_path in images:
        key = str(image_path.resolve())
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(image_path.resolve())
    return deduplicated


def parse_viewpoint_from_path(image_path: str | Path) -> str:
    match = VIEWPOINT_RE.search(Path(image_path).name)
    if match is None:
        raise ValueError(f"Could not parse viewpoint from {image_path}")
    return match.group(1).lower()


def viewpoint_histogram(image_paths: Sequence[Path]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for image_path in image_paths:
        view_id = parse_viewpoint_from_path(image_path)
        counts[view_id] = counts.get(view_id, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: viewpoint_sort_key(item[0])))


def write_image_list(path: Path, image_paths: Sequence[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for image_path in image_paths:
            handle.write(f"{image_path}\n")


def read_image_list(path: Path) -> list[Path]:
    with path.open("r", encoding="utf-8") as handle:
        return [Path(line.strip()) for line in handle if line.strip()]


def label_path_from_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    for index, part in enumerate(parts):
        if part == "images":
            parts[index] = "labels"
            return Path(*parts).with_suffix(".txt")
    raise ValueError(f"Could not derive label path from image path: {image_path}")


def control_manifest_path(experiment_root: Path) -> Path:
    return experiment_root / "manifests" / "matched_controls.csv"


def control_dir(experiment_root: Path, control_job: MatchedControlJob) -> Path:
    return experiment_root / "controls" / control_job.slug


def control_status_path(experiment_root: Path, control_job: MatchedControlJob) -> Path:
    return control_dir(experiment_root, control_job) / "status.json"


def control_metadata_path(experiment_root: Path, control_job: MatchedControlJob) -> Path:
    return control_dir(experiment_root, control_job) / "subset_metadata.json"


def load_control_jobs(path: Path) -> list[MatchedControlJob]:
    jobs: list[MatchedControlJob] = []
    for row in read_csv_rows(path):
        jobs.append(
            MatchedControlJob(
                control_index=int(row["control_index"]),
                control_id=str(row["control_id"]),
                label=str(row["label"]),
                source_group=str(row["source_group"]),
                source_id=str(row["source_id"]),
                source_label=str(row["source_label"]),
                train_count=int(row["train_count"]),
                val_count=int(row["val_count"]),
                seed=int(row["seed"]),
                sampling_strategy=str(row["sampling_strategy"]),
                reference_map50_95=float(row["reference_mAP50-95"]),
                reference_map50=float(row["reference_mAP50"]),
                reference_f1=float(row["reference_F1"]),
            )
        )
    return jobs


def control_job_by_id(path: Path, control_id: str) -> MatchedControlJob:
    for job in load_control_jobs(path):
        if job.control_id == control_id:
            return job
    raise KeyError(f"Unknown matched-control id: {control_id}")


def sample_images(image_paths: Sequence[Path], target_count: int, seed: int, strategy: str) -> list[Path]:
    if target_count < 0:
        raise ValueError("Target sample count must be non-negative.")

    deduplicated = sorted({str(path.resolve()): path.resolve() for path in image_paths}.values(), key=lambda item: str(item))
    if target_count > len(deduplicated):
        raise ValueError(f"Requested {target_count} images but only {len(deduplicated)} are available.")
    if target_count == len(deduplicated):
        return deduplicated
    if target_count == 0:
        return []

    if strategy == "global_random":
        rng = random.Random(stable_seed(f"global::{seed}"))
        selected = rng.sample(deduplicated, target_count)
        return sorted(selected, key=lambda item: str(item))

    if strategy != "stratified_viewpoint":
        raise ValueError(f"Unsupported sampling strategy: {strategy}")

    by_viewpoint: dict[str, list[Path]] = {}
    for image_path in deduplicated:
        by_viewpoint.setdefault(parse_viewpoint_from_path(image_path), []).append(image_path)

    total_images = len(deduplicated)
    viewpoints = sorted(by_viewpoint, key=viewpoint_sort_key)
    quotas: dict[str, int] = {}
    fractions: dict[str, float] = {}
    allocated = 0
    for view_id in viewpoints:
        exact = target_count * len(by_viewpoint[view_id]) / total_images
        base = min(len(by_viewpoint[view_id]), int(exact))
        quotas[view_id] = base
        fractions[view_id] = exact - base
        allocated += base

    remaining = target_count - allocated
    ranked_views = sorted(
        viewpoints,
        key=lambda view_id: (
            -fractions[view_id],
            stable_seed(f"fraction::{seed}::{view_id}"),
        ),
    )
    for view_id in ranked_views:
        if remaining <= 0:
            break
        if quotas[view_id] >= len(by_viewpoint[view_id]):
            continue
        quotas[view_id] += 1
        remaining -= 1

    if remaining > 0:
        spare_views = [view_id for view_id in ranked_views if quotas[view_id] < len(by_viewpoint[view_id])]
        if not spare_views:
            raise RuntimeError("Could not allocate the remaining quota for stratified sampling.")
        cursor = 0
        while remaining > 0:
            view_id = spare_views[cursor % len(spare_views)]
            if quotas[view_id] < len(by_viewpoint[view_id]):
                quotas[view_id] += 1
                remaining -= 1
            cursor += 1

    sampled: list[Path] = []
    for view_id in viewpoints:
        view_images = sorted(by_viewpoint[view_id], key=lambda item: str(item))
        quota = quotas[view_id]
        if quota <= 0:
            continue
        if quota >= len(view_images):
            sampled.extend(view_images)
            continue
        rng = random.Random(stable_seed(f"sample::{seed}::{view_id}"))
        sampled.extend(sorted(rng.sample(view_images, quota), key=lambda item: str(item)))

    if len(sampled) != target_count:
        raise RuntimeError(f"Sampled {len(sampled)} images but expected {target_count}.")
    return sorted(sampled, key=lambda item: (parse_viewpoint_from_path(item), item.name, str(item)))


def build_control_subset(
    base_data_yaml: Path,
    experiment_root: Path,
    control_job: MatchedControlJob,
    force: bool = False,
) -> dict[str, object]:
    ensure_control_experiment_root(experiment_root)
    job_dir = control_dir(experiment_root, control_job)
    metadata_path = control_metadata_path(experiment_root, control_job)
    if metadata_path.exists() and not force:
        metadata = read_json(metadata_path)
        validate_control_subset(metadata_path)
        return metadata

    data_dict = read_yaml(base_data_yaml)
    class_names = normalize_names(data_dict)
    dataset_root = resolve_dataset_root(base_data_yaml, data_dict)

    train_pool = resolve_split_images(base_data_yaml, "train")
    val_pool = resolve_split_images(base_data_yaml, "val")
    full_test_images = resolve_split_images(base_data_yaml, "test")
    train_images = sample_images(train_pool, control_job.train_count, control_job.seed, control_job.sampling_strategy)
    val_images = sample_images(val_pool, control_job.val_count, control_job.seed + 1_000_000, control_job.sampling_strategy)

    lists_dir = job_dir / "lists"
    train_list = lists_dir / "train.txt"
    val_list = lists_dir / "val.txt"
    test_full_list = lists_dir / "test_full.txt"
    write_image_list(train_list, train_images)
    write_image_list(val_list, val_images)
    write_image_list(test_full_list, full_test_images)

    data_yaml = job_dir / "data_control.yaml"
    write_yaml(
        data_yaml,
        {
            "path": str(dataset_root),
            "train": str(train_list.resolve()),
            "val": str(val_list.resolve()),
            "test": str(test_full_list.resolve()),
            "nc": len(class_names),
            "names": class_names,
        },
    )

    metadata: dict[str, object] = {
        "control_id": control_job.control_id,
        "label": control_job.label,
        "source_group": control_job.source_group,
        "source_id": control_job.source_id,
        "source_label": control_job.source_label,
        "seed": control_job.seed,
        "sampling_strategy": control_job.sampling_strategy,
        "reference_metrics": {
            "map50_95": control_job.reference_map50_95,
            "map50": control_job.reference_map50,
            "f1": control_job.reference_f1,
        },
        "control_dir": str(job_dir.resolve()),
        "base_data_yaml": str(base_data_yaml.resolve()),
        "dataset_root": str(dataset_root.resolve()),
        "class_names": class_names,
        "subset_files": {
            "train_list": str(train_list.resolve()),
            "val_list": str(val_list.resolve()),
            "test_full_list": str(test_full_list.resolve()),
            "data_yaml": str(data_yaml.resolve()),
        },
        "split_counts": {
            "train_images": len(train_images),
            "val_images": len(val_images),
            "test_full_images": len(full_test_images),
        },
        "viewpoint_counts": {
            "train": viewpoint_histogram(train_images),
            "val": viewpoint_histogram(val_images),
        },
        "missing_labels": {
            "train_missing_labels": sum(0 if label_path_from_image(path).exists() else 1 for path in train_images),
            "val_missing_labels": sum(0 if label_path_from_image(path).exists() else 1 for path in val_images),
        },
        "created_at": now_utc_iso(),
    }
    write_json_atomic(metadata_path, metadata)
    validate_control_subset(metadata_path)
    return metadata


def validate_control_subset(metadata_path: Path) -> dict[str, object]:
    metadata = read_json(metadata_path)
    subset_files = metadata["subset_files"]
    train_images = read_image_list(Path(subset_files["train_list"]))
    val_images = read_image_list(Path(subset_files["val_list"]))

    if len({str(path.resolve()) for path in train_images}) != len(train_images):
        raise ValueError("Duplicate train images detected in matched-control subset.")
    if len({str(path.resolve()) for path in val_images}) != len(val_images):
        raise ValueError("Duplicate val images detected in matched-control subset.")

    train_set = {str(path.resolve()) for path in train_images}
    val_set = {str(path.resolve()) for path in val_images}
    if train_set & val_set:
        raise ValueError("Train/val leakage detected in matched-control subset.")

    split_counts = metadata["split_counts"]
    expected_counts = {
        "train_images": len(train_images),
        "val_images": len(val_images),
    }
    for key, expected in expected_counts.items():
        if int(split_counts[key]) != expected:
            raise ValueError(f"Split count mismatch for {key}: expected {expected}, found {split_counts[key]}")

    return metadata


def load_baseline_summary_row(summary_csv: Path) -> dict[str, float | str] | None:
    if not summary_csv.exists():
        return None
    for row in read_csv_rows(summary_csv):
        if row.get("detector") == "YOLOv8l" and row.get("regime") == "M4":
            parsed: dict[str, float | str] = {"detector": row["detector"], "regime": row["regime"]}
            for key, value in row.items():
                if key in {"detector", "regime"}:
                    continue
                parsed[key] = float(value)
            return parsed
    return None


def default_training_args() -> dict[str, object]:
    return dict(_pair_lib().DEFAULT_TRAINING_ARGS)


def setup_yolo_environment(project_root: Path) -> None:
    _pair_lib().setup_yolo_environment(project_root)


def load_yolo():
    return _pair_lib().load_yolo()


def install_ultralytics_unique_label_cache(cache_dir: Path) -> None:
    _pair_lib().install_ultralytics_unique_label_cache(cache_dir)


def load_or_build_coco_gt_for_images(image_paths: Sequence[Path], class_names: Sequence[str], out_json: Path):
    return _pair_lib().load_or_build_coco_gt_for_images(image_paths, class_names, out_json)


def predict_yolo_to_coco_json(*args, **kwargs):
    return _pair_lib().predict_yolo_to_coco_json(*args, **kwargs)


def evaluate_coco(gt_json: Path, pred_json: Path) -> dict[str, float]:
    return _pair_lib().evaluate_coco(gt_json, pred_json)
