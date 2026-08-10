from __future__ import annotations

import csv
import importlib
import json
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
DEFAULT_FULL_BASELINE_SUMMARY = (
    Path("outputs") / "detector_family_comparison" / "standardized_test_eval" / "standardized_test_summary.csv"
)
DEFAULT_TRAINING_ARGS: dict[str, object] = {
    "model": "yolov8l.pt",
    "epochs": 100,
    "patience": 100,
    "batch": 16,
    "imgsz": 640,
    "save": True,
    "save_period": -1,
    "cache": False,
    "workers": 8,
    "pretrained": True,
    "optimizer": "auto",
    "verbose": True,
    "seed": 0,
    "deterministic": True,
    "single_cls": False,
    "rect": False,
    "cos_lr": False,
    "close_mosaic": 10,
    "amp": True,
    "fraction": 1.0,
    "profile": False,
    "multi_scale": 0.0,
    "compile": False,
    "dropout": 0.0,
    "val": True,
    "split": "val",
    "save_json": False,
    "iou": 0.7,
    "max_det": 300,
    "half": False,
    "dnn": False,
    "plots": False,
    "augment": False,
    "agnostic_nms": False,
    "retina_masks": False,
    "show": False,
    "save_frames": False,
    "save_txt": False,
    "save_conf": False,
    "save_crop": False,
    "show_labels": True,
    "show_conf": True,
    "show_boxes": True,
    "format": "torchscript",
    "keras": False,
    "optimize": False,
    "int8": False,
    "dynamic": False,
    "simplify": True,
    "nms": False,
    "lr0": 0.01,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1,
    "box": 7.5,
    "cls": 0.5,
    "dfl": 1.5,
    "pose": 12.0,
    "kobj": 1.0,
    "rle": 1.0,
    "angle": 1.0,
    "nbs": 64,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "bgr": 0.0,
    "mosaic": 0.0,
    "mixup": 0.0,
    "cutmix": 0.0,
    "copy_paste": 0.0,
    "copy_paste_mode": "flip",
    "auto_augment": "randaugment",
    "erasing": 0.4,
    "tracker": "botsort.yaml",
}

PAIR_LIB_DIR = Path(__file__).resolve().parents[1] / "m4_pair_subset_experiment"


def _pair_lib():
    if str(PAIR_LIB_DIR) not in sys.path:
        sys.path.insert(0, str(PAIR_LIB_DIR))
    return importlib.import_module("pair_experiment_lib")


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


SINGLE_PILOT_BLUEPRINTS: list[tuple[str, str, str]] = [
    ("low_far_front", "ellow-radfar-az000", "Low elevation and far radius reference viewpoint near the front azimuth."),
    ("low_mid_side", "ellow-radmid-az090", "Low elevation with a side azimuth to contrast against front-view pilots."),
    ("mid_mid_back", "elmid-radmid-az180", "Mid-elevation, mid-radius viewpoint to represent the central geometry band."),
    ("high_mid_side", "elhigh-radmid-az090", "High elevation side-viewpoint to test vertical separation against low pilots."),
    ("high_near_back", "elhigh-radnear-az270", "High and near viewpoint from the rear quadrant for geometric contrast."),
]


@dataclass(frozen=True)
class SingleJob:
    viewpoint_index: int
    single_id: str
    viewpoint: str
    pilot_rank: int | None = None
    pilot_name: str = ""
    pilot_note: str = ""

    @property
    def slug(self) -> str:
        return f"{self.single_id}__{safe_slug(self.viewpoint)}"


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_")


def parse_viewpoint(view_id: str) -> tuple[str, str, int]:
    match = VIEWPOINT_PARTS_RE.match(view_id.strip().lower())
    if match is None:
        raise ValueError(f"Could not parse viewpoint id: {view_id}")
    elevation, radius, azimuth = match.groups()
    return elevation, radius, int(azimuth)


def viewpoint_sort_key(view_id: str) -> tuple[int, int, int]:
    elevation, radius, azimuth = parse_viewpoint(view_id)
    return (ELEVATION_SORT.get(elevation, 99), RADIUS_SORT.get(radius, 99), azimuth)


def human_viewpoint_label(view_id: str) -> str:
    elevation, radius, azimuth = parse_viewpoint(view_id)
    elevation_map = {"ellow": "low", "elmid": "mid", "elhigh": "high"}
    radius_map = {"radnear": "near", "radmid": "mid", "radfar": "far"}
    return f"{elevation_map.get(elevation, elevation)} | {radius_map.get(radius, radius)} | az{azimuth:03d}"


def short_viewpoint_label(view_id: str) -> str:
    elevation, radius, azimuth = parse_viewpoint(view_id)
    elevation_map = {"ellow": "L", "elmid": "M", "elhigh": "H"}
    radius_map = {"radnear": "N", "radmid": "M", "radfar": "F"}
    return f"{elevation_map.get(elevation, '?')}-{radius_map.get(radius, '?')}-{azimuth:03d}"


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


def ensure_single_experiment_root(experiment_root: Path) -> None:
    for relative in ("manifests", "singles", "shared", "plots", "reports", "launchers", "logs"):
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
        return [str(name) for _, name in sorted(((int(key), value) for key, value in names.items()), key=lambda item: item[0])]
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
                    images.append(image_path)
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
                        images.append(listed_path)
            continue
        if resolved.is_file() and resolved.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(resolved)
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


def viewpoint_counts_by_split(data_yaml: Path, splits: Sequence[str] = ("train", "val", "test")) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for split in splits:
        counter: dict[str, int] = {}
        for image_path in resolve_split_images(data_yaml, split):
            view_id = parse_viewpoint_from_path(image_path)
            counter[view_id] = counter.get(view_id, 0) + 1
        counts[split] = counter
    return counts


def collect_all_viewpoints(data_yaml: Path, splits: Sequence[str] = ("train", "val", "test")) -> list[str]:
    views: set[str] = set()
    for split in splits:
        for image_path in resolve_split_images(data_yaml, split):
            views.add(parse_viewpoint_from_path(image_path))
    return sorted(views, key=viewpoint_sort_key)


def enumerate_single_jobs(viewpoints: Sequence[str], pilot_count: int = 5) -> list[SingleJob]:
    sorted_views = sorted(viewpoints, key=viewpoint_sort_key)
    jobs: list[SingleJob] = []
    pilot_blueprints = SINGLE_PILOT_BLUEPRINTS[: max(0, pilot_count)]
    pilot_map = {viewpoint: (index + 1, name, note) for index, (name, viewpoint, note) in enumerate(pilot_blueprints)}

    for index, viewpoint in enumerate(sorted_views, start=1):
        pilot_info = pilot_map.get(viewpoint)
        jobs.append(
            SingleJob(
                viewpoint_index=index,
                single_id=f"sv{index:04d}",
                viewpoint=viewpoint,
                pilot_rank=None if pilot_info is None else pilot_info[0],
                pilot_name="" if pilot_info is None else pilot_info[1],
                pilot_note="" if pilot_info is None else pilot_info[2],
            )
        )
    return jobs


def single_manifest_path(experiment_root: Path) -> Path:
    return experiment_root / "manifests" / "full_single_viewpoints.csv"


def load_single_jobs(path: Path) -> list[SingleJob]:
    jobs: list[SingleJob] = []
    for row in read_csv_rows(path):
        pilot_rank_raw = str(row.get("pilot_rank", "")).strip()
        jobs.append(
            SingleJob(
                viewpoint_index=int(row["viewpoint_index"]),
                single_id=str(row["single_id"]),
                viewpoint=str(row["viewpoint"]),
                pilot_rank=None if not pilot_rank_raw else int(pilot_rank_raw),
                pilot_name=str(row.get("pilot_name", "")),
                pilot_note=str(row.get("pilot_note", "")),
            )
        )
    return jobs


def single_job_by_id(path: Path, single_id: str) -> SingleJob:
    for job in load_single_jobs(path):
        if job.single_id == single_id:
            return job
    raise KeyError(f"Unknown single-viewpoint id: {single_id}")


def single_dir(experiment_root: Path, single_job: SingleJob) -> Path:
    return experiment_root / "singles" / single_job.slug


def single_status_path(experiment_root: Path, single_job: SingleJob) -> Path:
    return single_dir(experiment_root, single_job) / "status.json"


def single_metadata_path(experiment_root: Path, single_job: SingleJob) -> Path:
    return single_dir(experiment_root, single_job) / "subset_metadata.json"


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


def filter_images_by_viewpoints(image_paths: Sequence[Path], allowed_viewpoints: set[str]) -> list[Path]:
    filtered = [path for path in image_paths if parse_viewpoint_from_path(path) in allowed_viewpoints]
    return sorted(filtered, key=lambda item: (parse_viewpoint_from_path(item), item.name))


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


def build_single_subset(
    base_data_yaml: Path,
    experiment_root: Path,
    single_job: SingleJob,
    force: bool = False,
) -> dict[str, object]:
    ensure_single_experiment_root(experiment_root)
    job_dir = single_dir(experiment_root, single_job)
    metadata_path = single_metadata_path(experiment_root, single_job)
    if metadata_path.exists() and not force:
        metadata = read_json(metadata_path)
        validate_single_subset(metadata_path)
        return metadata

    data_dict = read_yaml(base_data_yaml)
    class_names = normalize_names(data_dict)
    dataset_root = resolve_dataset_root(base_data_yaml, data_dict)
    allowed = {single_job.viewpoint}

    train_images = filter_images_by_viewpoints(resolve_split_images(base_data_yaml, "train"), allowed)
    val_images = filter_images_by_viewpoints(resolve_split_images(base_data_yaml, "val"), allowed)
    test_view_images = filter_images_by_viewpoints(resolve_split_images(base_data_yaml, "test"), allowed)
    full_test_images = resolve_split_images(base_data_yaml, "test")

    lists_dir = job_dir / "lists"
    train_list = lists_dir / "train.txt"
    val_list = lists_dir / "val.txt"
    test_view_list = lists_dir / "test_viewpoint.txt"
    write_image_list(train_list, train_images)
    write_image_list(val_list, val_images)
    write_image_list(test_view_list, test_view_images)

    single_yaml = job_dir / "data_single.yaml"
    write_yaml(
        single_yaml,
        {
            "path": str(dataset_root),
            "train": str(train_list.resolve()),
            "val": str(val_list.resolve()),
            "test": str(test_view_list.resolve()),
            "nc": len(class_names),
            "names": class_names,
        },
    )

    missing_labels = {
        "train_missing_labels": sum(0 if label_path_from_image(path).exists() else 1 for path in train_images),
        "val_missing_labels": sum(0 if label_path_from_image(path).exists() else 1 for path in val_images),
        "test_view_missing_labels": sum(0 if label_path_from_image(path).exists() else 1 for path in test_view_images),
    }

    metadata: dict[str, object] = {
        "single_id": single_job.single_id,
        "viewpoint_index": single_job.viewpoint_index,
        "viewpoint": single_job.viewpoint,
        "single_dir": str(job_dir.resolve()),
        "base_data_yaml": str(base_data_yaml.resolve()),
        "dataset_root": str(dataset_root.resolve()),
        "class_names": class_names,
        "subset_files": {
            "train_list": str(train_list.resolve()),
            "val_list": str(val_list.resolve()),
            "test_view_list": str(test_view_list.resolve()),
            "data_yaml": str(single_yaml.resolve()),
        },
        "split_counts": {
            "train_images": len(train_images),
            "val_images": len(val_images),
            "test_view_images": len(test_view_images),
            "test_full_images": len(full_test_images),
        },
        "pilot_rank": single_job.pilot_rank,
        "pilot_name": single_job.pilot_name,
        "pilot_note": single_job.pilot_note,
        "missing_labels": missing_labels,
        "created_at": now_utc_iso(),
    }
    write_json_atomic(metadata_path, metadata)
    validate_single_subset(metadata_path)
    return metadata


def validate_single_subset(metadata_path: Path) -> dict[str, object]:
    metadata = read_json(metadata_path)
    allowed = {str(metadata["viewpoint"])}
    subset_files = metadata["subset_files"]
    train_images = read_image_list(Path(subset_files["train_list"]))
    val_images = read_image_list(Path(subset_files["val_list"]))
    test_view_images = read_image_list(Path(subset_files["test_view_list"]))

    for split_name, image_paths in (("train", train_images), ("val", val_images), ("test_view", test_view_images)):
        for image_path in image_paths:
            if parse_viewpoint_from_path(image_path) not in allowed:
                raise ValueError(f"{split_name} subset contains an out-of-viewpoint image: {image_path}")

    train_set = {str(path) for path in train_images}
    val_set = {str(path) for path in val_images}
    test_view_set = {str(path) for path in test_view_images}
    if train_set & val_set:
        raise ValueError("Train/val leakage detected in single-viewpoint subset.")
    if train_set & test_view_set:
        raise ValueError("Train/test leakage detected in single-viewpoint subset.")
    if val_set & test_view_set:
        raise ValueError("Val/test leakage detected in single-viewpoint subset.")

    split_counts = metadata["split_counts"]
    expected_counts = {
        "train_images": len(train_images),
        "val_images": len(val_images),
        "test_view_images": len(test_view_images),
    }
    for key, expected in expected_counts.items():
        if int(split_counts[key]) != expected:
            raise ValueError(f"Split count mismatch for {key}: expected {expected}, found {split_counts[key]}")

    return metadata
