from __future__ import annotations

import csv
import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import yaml
from PIL import Image

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except ImportError as exc:
    raise SystemExit(
        "pycocotools is required for the M4 pair-subset experiment. "
        "Install it in the project environment before running the evaluation pipeline."
    ) from exc


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIEWPOINT_RE = re.compile(r"-(el[a-z]+-rad[a-z]+-az\d+)(?=\.[^.]+$)", re.IGNORECASE)
VIEWPOINT_PARTS_RE = re.compile(r"^(el[a-z]+)-(rad[a-z]+)-az(\d+)$", re.IGNORECASE)
ELEVATION_SORT = {"ellow": 0, "elmid": 1, "elhigh": 2}
RADIUS_SORT = {"radnear": 0, "radmid": 1, "radfar": 2}
DEFAULT_FULL_BASELINE_SUMMARY = (
    Path("outputs")
    / "detector_family_comparison"
    / "standardized_test_eval"
    / "standardized_test_summary.csv"
)
DEFAULT_PROTOCOL_RECOMMENDATION = "option_a_full_test"
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
PILOT_BLUEPRINTS: list[tuple[str, str, str, str]] = [
    (
        "redundant_neighbor",
        "ellow-radfar-az000",
        "ellow-radfar-az045",
        "Two nearby low/far azimuths to test a deliberately redundant pair.",
    ),
    (
        "elevation_only",
        "ellow-radmid-az090",
        "elhigh-radmid-az090",
        "Same azimuth/radius, different elevation, to isolate vertical diversity.",
    ),
    (
        "radius_only",
        "elhigh-radnear-az270",
        "elhigh-radfar-az270",
        "Same azimuth/elevation, different radius, to isolate distance diversity.",
    ),
    (
        "azimuth_only",
        "elmid-radmid-az000",
        "elmid-radmid-az180",
        "Same elevation/radius, opposite azimuths, to isolate azimuth diversity.",
    ),
    (
        "max_contrast",
        "ellow-radfar-az000",
        "elhigh-radnear-az180",
        "Two geometrically distant viewpoints to stress maximal training diversity.",
    ),
]


@dataclass(frozen=True)
class PairJob:
    pair_index: int
    pair_id: str
    viewpoint_1: str
    viewpoint_2: str
    pilot_rank: int | None = None
    pilot_name: str = ""
    pilot_note: str = ""

    @property
    def slug(self) -> str:
        return f"{self.pair_id}__{safe_slug(self.viewpoint_1)}__{safe_slug(self.viewpoint_2)}"

    @property
    def viewpoints(self) -> tuple[str, str]:
        return self.viewpoint_1, self.viewpoint_2


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
    return (
        ELEVATION_SORT.get(elevation, 99),
        RADIUS_SORT.get(radius, 99),
        azimuth,
    )


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


def ensure_experiment_root(experiment_root: Path) -> None:
    for relative in ("manifests", "pairs", "shared", "plots", "reports", "launchers"):
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


def enumerate_pair_jobs(viewpoints: Sequence[str], pilot_count: int = 5) -> list[PairJob]:
    selected_pilots: dict[tuple[str, str], tuple[int, str, str]] = {}
    for rank, (pilot_name, first, second, note) in enumerate(PILOT_BLUEPRINTS[: max(0, pilot_count)], start=1):
        ordered = tuple(sorted((first, second), key=viewpoint_sort_key))
        if ordered[0] in viewpoints and ordered[1] in viewpoints:
            selected_pilots[ordered] = (rank, pilot_name, note)

    jobs: list[PairJob] = []
    for pair_index, (first, second) in enumerate(combinations(viewpoints, 2), start=1):
        ordered = tuple(sorted((first, second), key=viewpoint_sort_key))
        pilot_meta = selected_pilots.get(ordered)
        jobs.append(
            PairJob(
                pair_index=pair_index,
                pair_id=f"p{pair_index:04d}",
                viewpoint_1=ordered[0],
                viewpoint_2=ordered[1],
                pilot_rank=pilot_meta[0] if pilot_meta else None,
                pilot_name=pilot_meta[1] if pilot_meta else "",
                pilot_note=pilot_meta[2] if pilot_meta else "",
            )
        )

    if len(selected_pilots) < pilot_count:
        next_rank = len(selected_pilots) + 1
        for index, job in enumerate(jobs):
            if job.pilot_rank is not None:
                continue
            if next_rank > pilot_count:
                break
            jobs[index] = PairJob(
                pair_index=job.pair_index,
                pair_id=job.pair_id,
                viewpoint_1=job.viewpoint_1,
                viewpoint_2=job.viewpoint_2,
                pilot_rank=next_rank,
                pilot_name="fallback",
                pilot_note="Fallback pilot pair added because a blueprint viewpoint was unavailable.",
            )
            next_rank += 1

    return jobs


def pair_manifest_path(experiment_root: Path) -> Path:
    return experiment_root / "manifests" / "viewpoint_pairs.csv"


def load_pair_jobs(path: Path) -> list[PairJob]:
    rows = read_csv_rows(path)
    jobs: list[PairJob] = []
    for row in rows:
        pilot_rank_raw = row.get("pilot_rank", "").strip()
        jobs.append(
            PairJob(
                pair_index=int(row["pair_index"]),
                pair_id=row["pair_id"],
                viewpoint_1=row["viewpoint_1"],
                viewpoint_2=row["viewpoint_2"],
                pilot_rank=int(pilot_rank_raw) if pilot_rank_raw else None,
                pilot_name=row.get("pilot_name", ""),
                pilot_note=row.get("pilot_note", ""),
            )
        )
    return jobs


def pair_job_by_id(path: Path, pair_id: str) -> PairJob:
    for job in load_pair_jobs(path):
        if job.pair_id == pair_id:
            return job
    raise KeyError(f"Pair id '{pair_id}' was not found in {path}.")


def pair_dir(experiment_root: Path, pair_job: PairJob) -> Path:
    return experiment_root / "pairs" / pair_job.slug


def pair_status_path(experiment_root: Path, pair_job: PairJob) -> Path:
    return pair_dir(experiment_root, pair_job) / "status.json"


def pair_metadata_path(experiment_root: Path, pair_job: PairJob) -> Path:
    return pair_dir(experiment_root, pair_job) / "subset_metadata.json"


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


def build_pair_subset(
    base_data_yaml: Path,
    experiment_root: Path,
    pair_job: PairJob,
    force: bool = False,
) -> dict[str, object]:
    ensure_experiment_root(experiment_root)
    job_dir = pair_dir(experiment_root, pair_job)
    metadata_path = job_dir / "subset_metadata.json"
    if metadata_path.exists() and not force:
        metadata = read_json(metadata_path)
        validate_pair_subset(metadata_path)
        return metadata

    data_dict = read_yaml(base_data_yaml)
    class_names = normalize_names(data_dict)
    dataset_root = resolve_dataset_root(base_data_yaml, data_dict)
    allowed = {pair_job.viewpoint_1, pair_job.viewpoint_2}

    train_images = filter_images_by_viewpoints(resolve_split_images(base_data_yaml, "train"), allowed)
    val_images = filter_images_by_viewpoints(resolve_split_images(base_data_yaml, "val"), allowed)
    test_pair_images = filter_images_by_viewpoints(resolve_split_images(base_data_yaml, "test"), allowed)
    full_test_images = resolve_split_images(base_data_yaml, "test")

    lists_dir = job_dir / "lists"
    train_list = lists_dir / "train.txt"
    val_list = lists_dir / "val.txt"
    test_pair_list = lists_dir / "test_pair.txt"
    write_image_list(train_list, train_images)
    write_image_list(val_list, val_images)
    write_image_list(test_pair_list, test_pair_images)

    pair_yaml = job_dir / "data_pair.yaml"
    write_yaml(
        pair_yaml,
        {
            "path": str(dataset_root),
            "train": str(train_list.resolve()),
            "val": str(val_list.resolve()),
            "test": str(test_pair_list.resolve()),
            "nc": len(class_names),
            "names": class_names,
        },
    )

    missing_labels = {
        "train_missing_labels": sum(0 if label_path_from_image(path).exists() else 1 for path in train_images),
        "val_missing_labels": sum(0 if label_path_from_image(path).exists() else 1 for path in val_images),
        "test_pair_missing_labels": sum(0 if label_path_from_image(path).exists() else 1 for path in test_pair_images),
    }

    metadata: dict[str, object] = {
        "pair_id": pair_job.pair_id,
        "pair_index": pair_job.pair_index,
        "viewpoint_1": pair_job.viewpoint_1,
        "viewpoint_2": pair_job.viewpoint_2,
        "pair_dir": str(job_dir.resolve()),
        "base_data_yaml": str(base_data_yaml.resolve()),
        "dataset_root": str(dataset_root.resolve()),
        "class_names": class_names,
        "subset_files": {
            "train_list": str(train_list.resolve()),
            "val_list": str(val_list.resolve()),
            "test_pair_list": str(test_pair_list.resolve()),
            "data_yaml": str(pair_yaml.resolve()),
        },
        "split_counts": {
            "train_images": len(train_images),
            "val_images": len(val_images),
            "test_pair_images": len(test_pair_images),
            "test_full_images": len(full_test_images),
        },
        "pilot_rank": pair_job.pilot_rank,
        "pilot_name": pair_job.pilot_name,
        "pilot_note": pair_job.pilot_note,
        "missing_labels": missing_labels,
        "created_at": now_utc_iso(),
    }
    write_json_atomic(metadata_path, metadata)
    validate_pair_subset(metadata_path)
    return metadata


def validate_pair_subset(metadata_path: Path) -> dict[str, object]:
    metadata = read_json(metadata_path)
    allowed = {str(metadata["viewpoint_1"]), str(metadata["viewpoint_2"])}
    subset_files = metadata["subset_files"]
    train_images = read_image_list(Path(subset_files["train_list"]))
    val_images = read_image_list(Path(subset_files["val_list"]))
    test_pair_images = read_image_list(Path(subset_files["test_pair_list"]))

    for split_name, image_paths in (
        ("train", train_images),
        ("val", val_images),
        ("test_pair", test_pair_images),
    ):
        for image_path in image_paths:
            if parse_viewpoint_from_path(image_path) not in allowed:
                raise ValueError(f"{split_name} subset contains an out-of-pair image: {image_path}")

    train_set = {str(path) for path in train_images}
    val_set = {str(path) for path in val_images}
    test_pair_set = {str(path) for path in test_pair_images}
    if train_set & val_set:
        raise ValueError("Train/val leakage detected in pair subset.")
    if train_set & test_pair_set:
        raise ValueError("Train/test leakage detected in pair subset.")
    if val_set & test_pair_set:
        raise ValueError("Val/test leakage detected in pair subset.")

    split_counts = metadata["split_counts"]
    expected_counts = {
        "train_images": len(train_images),
        "val_images": len(val_images),
        "test_pair_images": len(test_pair_images),
    }
    for key, expected in expected_counts.items():
        if int(split_counts[key]) != expected:
            raise ValueError(f"Metadata count mismatch for {key}: {split_counts[key]} vs {expected}")

    return metadata


def setup_yolo_environment(project_root: Path) -> None:
    os.environ.setdefault("YOLO_CONFIG_DIR", str((project_root / "Ultralytics").resolve()))


def load_yolo() -> type:
    from ultralytics import YOLO

    return YOLO


def install_ultralytics_unique_label_cache(cache_dir: Path) -> None:
    """Patch Ultralytics label caching so each pair job uses its own cache files.

    Ultralytics normally writes one cache next to the shared label directory
    (for example ``train_M4.cache`` and ``val.cache``). When many Slurm array
    tasks start in parallel on the same dataset root, they contend on those
    shared files and can fail with unlink or stale-handle errors on networked
    storage. This patch redirects label caches into a per-job directory keyed
    by the exact image subset hash.
    """

    cache_dir = cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    from ultralytics.data import dataset as dataset_module

    existing = getattr(dataset_module.YOLODataset, "_m4_pair_cache_dir", "")
    if existing == str(cache_dir):
        return

    from ultralytics.utils import LOCAL_RANK, LOGGER, TQDM

    data_cache_version = dataset_module.DATASET_CACHE_VERSION
    help_url = dataset_module.HELP_URL
    get_hash = dataset_module.get_hash
    img2label_paths = dataset_module.img2label_paths
    load_dataset_cache_file = dataset_module.load_dataset_cache_file

    def patched_save_dataset_cache_file(prefix: str, path: Path, payload: dict, version: str) -> None:
        path = Path(path)
        payload["version"] = version
        tmp_path = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
        try:
            with tmp_path.open("wb") as handle:
                np.save(handle, payload)
            tmp_path.replace(path)
            LOGGER.info(f"{prefix}New cache created: {path}")
        except Exception as exc:  # pragma: no cover - runtime protection
            tmp_path.unlink(missing_ok=True)
            LOGGER.warning(f"{prefix}WARNING: failed to save cache to {path}: {exc}")

    def patched_get_labels(self) -> list[dict]:
        self.label_files = img2label_paths(self.im_files)
        subset_hash = get_hash(self.label_files + self.im_files)
        split_name = Path(self.im_files[0]).parent.name if self.im_files else "dataset"
        cache_path = cache_dir / f"{split_name}_{subset_hash[:16]}.cache"

        try:
            cache, exists = load_dataset_cache_file(cache_path), True
            assert cache["version"] == data_cache_version
            assert cache["hash"] == subset_hash
        except (FileNotFoundError, AssertionError, AttributeError, ModuleNotFoundError, OSError):
            cache, exists = self.cache_labels(cache_path), False

        nf, nm, ne, nc, n = cache.pop("results")
        if exists and LOCAL_RANK in {-1, 0}:
            desc = f"Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            TQDM(None, desc=self.prefix + desc, total=n, initial=n)
            if cache["msgs"]:
                LOGGER.info("\n".join(cache["msgs"]))

        for key in ("hash", "version", "msgs"):
            cache.pop(key, None)
        labels = cache["labels"]
        if not labels:
            raise RuntimeError(
                f"No valid images found in {cache_path}. Images with incorrectly formatted labels are ignored. {help_url}"
            )

        self.im_files = [label["im_file"] for label in labels]
        lengths = ((len(label["cls"]), len(label["bboxes"]), len(label["segments"])) for label in labels)
        len_cls, len_boxes, len_segments = (sum(values) for values in zip(*lengths))
        if len_segments and len_boxes != len_segments:
            LOGGER.warning(
                f"Box and segment counts should be equal, but got len(segments) = {len_segments}, "
                f"len(boxes) = {len_boxes}. To resolve this only boxes will be used and all segments will be removed. "
                "To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset."
            )
            for label in labels:
                label["segments"] = []
        if len_cls == 0:
            LOGGER.warning(f"Labels are missing or empty in {cache_path}, training may not work correctly. {help_url}")
        return labels

    dataset_module.save_dataset_cache_file = patched_save_dataset_cache_file
    dataset_module.YOLODataset.get_labels = patched_get_labels
    dataset_module.YOLODataset._m4_pair_cache_dir = str(cache_dir)


def iter_chunks(values: Sequence[Path], chunk_size: int) -> Iterable[Sequence[Path]]:
    for start in range(0, len(values), chunk_size):
        yield values[start : start + chunk_size]


def image_size(image_path: Path) -> tuple[int, int]:
    with Image.open(image_path) as image:
        return image.size


def load_yolo_annotations(label_path: Path, width: int, height: int) -> list[dict[str, float | int]]:
    if not label_path.exists():
        return []

    annotations: list[dict[str, float | int]] = []
    with label_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            class_id, xc, yc, w, h = parts
            class_id = int(float(class_id))
            xc = float(xc) * width
            yc = float(yc) * height
            w = float(w) * width
            h = float(h) * height
            x = xc - w / 2
            y = yc - h / 2
            annotations.append(
                {
                    "category_id": class_id + 1,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                }
            )
    return annotations


def build_coco_gt_for_images(image_paths: Sequence[Path], class_names: Sequence[str], out_json: Path) -> tuple[list[Path], dict[str, int]]:
    coco_images: list[dict[str, object]] = []
    coco_annotations: list[dict[str, object]] = []
    image_id_map: dict[str, int] = {}
    annotation_id = 1

    categories = [{"id": idx + 1, "name": str(name)} for idx, name in enumerate(class_names)]

    for image_id, image_path in enumerate(sorted(image_paths), start=1):
        width, height = image_size(image_path)
        image_id_map[str(image_path)] = image_id
        coco_images.append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": width,
                "height": height,
            }
        )

        label_path = label_path_from_image(image_path)
        for annotation in load_yolo_annotations(label_path, width=width, height=height):
            coco_annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    **annotation,
                }
            )
            annotation_id += 1

    payload = {"images": coco_images, "annotations": coco_annotations, "categories": categories}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload), encoding="utf-8")
    return sorted(image_paths), image_id_map


def load_or_build_coco_gt_for_images(
    image_paths: Sequence[Path],
    class_names: Sequence[str],
    out_json: Path,
) -> tuple[list[Path], dict[str, int]]:
    ordered_paths = sorted(image_paths)
    image_id_map = {str(path): index for index, path in enumerate(ordered_paths, start=1)}
    if out_json.exists():
        return ordered_paths, image_id_map
    return build_coco_gt_for_images(ordered_paths, class_names, out_json)


def predict_yolo_to_coco_json(
    model_path: Path,
    image_paths: Sequence[Path],
    image_id_map: dict[str, int],
    out_json: Path,
    imgsz: int = 640,
    conf: float = 0.001,
    batch: int = 16,
    device: str | None = None,
    project_root: Path | None = None,
) -> None:
    if project_root is not None:
        setup_yolo_environment(project_root)
    YOLO = load_yolo()
    model = YOLO(str(model_path))
    results_json: list[dict[str, object]] = []

    for batch_paths in iter_chunks(list(image_paths), batch):
        results = model.predict(
            source=[str(path) for path in batch_paths],
            imgsz=imgsz,
            conf=conf,
            device=device,
            verbose=False,
        )
        for image_path, result in zip(batch_paths, results):
            if result.boxes is None or len(result.boxes) == 0:
                continue
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            for box, score, class_id in zip(boxes, scores, classes):
                x1, y1, x2, y2 = box.tolist()
                results_json.append(
                    {
                        "image_id": image_id_map[str(image_path)],
                        "category_id": int(class_id) + 1,
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": float(score),
                    }
                )

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results_json), encoding="utf-8")


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else float("nan")


def coco_precision_recall_f1(ev: COCOeval) -> tuple[float, float, float]:
    iou_thresholds = ev.params.iouThrs
    iou_index = int(np.argmin(np.abs(iou_thresholds - 0.5)))
    precision = ev.eval["precision"][iou_index, :, :, 0, -1]
    recall = ev.eval["recall"][iou_index, :, 0, -1]

    precision_values = precision[precision > -1]
    recall_values = recall[recall > -1]
    precision_50 = float(np.mean(precision_values)) if precision_values.size else float("nan")
    recall_50 = float(np.mean(recall_values)) if recall_values.size else float("nan")
    f1_50 = safe_divide(2 * precision_50 * recall_50, precision_50 + recall_50)
    return precision_50, recall_50, f1_50


def coco_bbox_to_xyxy(bbox: list[float]) -> tuple[float, float, float, float]:
    x, y, w, h = bbox
    return x, y, x + w, y + h


def iou_xyxy(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def matched_mean_iou(gt_json: Path, pred_json: Path, score_threshold: float = 0.001) -> float:
    gt = read_json(gt_json)
    preds = read_json(pred_json)

    gt_by_image: dict[int, list[dict]] = {}
    for annotation in gt["annotations"]:
        gt_by_image.setdefault(int(annotation["image_id"]), []).append(annotation)

    pred_by_image: dict[int, list[dict]] = {}
    for prediction in preds:
        if float(prediction["score"]) < score_threshold:
            continue
        pred_by_image.setdefault(int(prediction["image_id"]), []).append(prediction)

    matched_ious: list[float] = []
    for image_id, image_predictions in pred_by_image.items():
        image_annotations = gt_by_image.get(image_id, [])
        sorted_predictions = sorted(image_predictions, key=lambda item: float(item["score"]), reverse=True)
        matched_gt_ids: set[int] = set()

        for prediction in sorted_predictions:
            prediction_box = coco_bbox_to_xyxy(prediction["bbox"])
            prediction_class = int(prediction["category_id"])
            best_iou = 0.0
            best_gt_id = None
            for annotation in image_annotations:
                annotation_id = int(annotation["id"])
                if annotation_id in matched_gt_ids or int(annotation["category_id"]) != prediction_class:
                    continue
                annotation_box = coco_bbox_to_xyxy(annotation["bbox"])
                current_iou = iou_xyxy(prediction_box, annotation_box)
                if current_iou >= 0.5 and current_iou > best_iou:
                    best_iou = current_iou
                    best_gt_id = annotation_id
            if best_gt_id is not None:
                matched_gt_ids.add(best_gt_id)
                matched_ious.append(best_iou)

    return float(np.mean(matched_ious)) if matched_ious else float("nan")


def evaluate_coco(gt_json: Path, pred_json: Path) -> dict[str, float]:
    coco_gt = COCO(str(gt_json))
    coco_dt = coco_gt.loadRes(str(pred_json))
    evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    stats = evaluator.stats
    precision_50, recall_50, f1_50 = coco_precision_recall_f1(evaluator)
    return {
        "precision": precision_50,
        "recall": recall_50,
        "f1": f1_50,
        "map50": float(stats[1]),
        "map50_95": float(stats[0]),
        "ap75": float(stats[2]),
        "matched_mean_iou": matched_mean_iou(gt_json, pred_json),
    }


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
