#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import shutil
from collections import Counter
from pathlib import Path

import yaml


TARGET_NAMES = [
    "tent",
    "tank",
    "tower",
    "container",
    "whitevan",
    "suv",
    "male",
    "rock",
    "barrel",
    "tree",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def normalize(value: object) -> str:
    return str(value).strip().lower()


def load_names(yaml_path: Path) -> dict[int, str]:
    with yaml_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    names = config.get("names")
    if isinstance(names, list):
        return {i: normalize(name) for i, name in enumerate(names)}
    if isinstance(names, dict):
        return {int(i): normalize(name) for i, name in names.items()}
    raise ValueError(f"Could not interpret 'names' in {yaml_path}")


def convert_label(
    source_label: Path,
    destination_label: Path,
    old_to_new: dict[int, int],
) -> tuple[Counter, int, int, int]:
    class_counts: Counter = Counter()
    output_lines: list[str] = []
    seen: set[tuple[int, float, float, float, float]] = set()

    boxes = 0
    polygons = 0
    duplicates = 0

    for line_number, raw_line in enumerate(
        source_label.read_text(encoding="utf-8").splitlines(), start=1
    ):
        parts = raw_line.strip().split()
        if not parts:
            continue

        try:
            old_id_float = float(parts[0])
            old_id = int(old_id_float)
        except ValueError as exc:
            raise ValueError(
                f"Invalid class ID in {source_label}, line {line_number}: {raw_line}"
            ) from exc

        if old_id_float != old_id:
            raise ValueError(
                f"Non-integer class ID in {source_label}, line {line_number}"
            )
        if old_id not in old_to_new:
            raise ValueError(
                f"Unknown class ID {old_id} in {source_label}, line {line_number}"
            )

        new_id = old_to_new[old_id]

        try:
            coordinates = [float(value) for value in parts[1:]]
        except ValueError as exc:
            raise ValueError(
                f"Invalid coordinates in {source_label}, line {line_number}"
            ) from exc

        if len(coordinates) == 4:
            x_center, y_center, width, height = coordinates
            boxes += 1
        elif len(coordinates) >= 6 and len(coordinates) % 2 == 0:
            xs = coordinates[0::2]
            ys = coordinates[1::2]

            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

            x_center = (x_min + x_max) / 2.0
            y_center = (y_min + y_max) / 2.0
            width = x_max - x_min
            height = y_max - y_min
            polygons += 1
        else:
            raise ValueError(
                f"Unsupported annotation format in {source_label}, "
                f"line {line_number}: {len(parts)} values"
            )

        values = [x_center, y_center, width, height]

        if any(value < 0.0 or value > 1.0 for value in values):
            raise ValueError(
                f"Box outside normalized range in {source_label}, "
                f"line {line_number}: {values}"
            )
        if width <= 0.0 or height <= 0.0:
            raise ValueError(
                f"Zero-sized box in {source_label}, line {line_number}"
            )

        key = (
            new_id,
            round(x_center, 8),
            round(y_center, 8),
            round(width, 8),
            round(height, 8),
        )

        if key in seen:
            duplicates += 1
            continue

        seen.add(key)
        class_counts[new_id] += 1

        output_lines.append(
            f"{new_id} {x_center:.8f} {y_center:.8f} "
            f"{width:.8f} {height:.8f}"
        )

    destination_label.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(output_lines)
    if text:
        text += "\n"
    destination_label.write_text(text, encoding="utf-8")

    return class_counts, boxes, polygons, duplicates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source_root = args.source.resolve()
    output_root = args.output.resolve()
    source_yaml = source_root / "data.yaml"

    if not source_yaml.is_file():
        raise FileNotFoundError(f"Missing source YAML: {source_yaml}")
    if output_root.exists():
        raise FileExistsError(
            f"Output already exists: {output_root}\n"
            "Choose a new output path or move the old output first."
        )

    source_names = load_names(source_yaml)
    target_name_to_id = {
        class_name: class_id
        for class_id, class_name in enumerate(TARGET_NAMES)
    }

    if set(source_names.values()) != set(TARGET_NAMES):
        raise ValueError(
            "Source and target class-name sets differ.\n"
            f"Source: {source_names}\nTarget: {TARGET_NAMES}"
        )

    old_to_new = {
        old_id: target_name_to_id[class_name]
        for old_id, class_name in source_names.items()
    }

    print("Class remapping:")
    for old_id in sorted(old_to_new):
        print(
            f"  {old_id}: {source_names[old_id]} -> {old_to_new[old_id]}"
        )

    split_sources = {
        "train": source_root / "train",
        "val": (
            source_root / "val"
            if (source_root / "val").is_dir()
            else source_root / "valid"
        ),
        "test": source_root / "test",
    }

    for split, split_root in split_sources.items():
        if not split_root.is_dir():
            raise FileNotFoundError(
                f"Missing required {split} split: expected {split_root}"
            )
        if not (split_root / "images").is_dir():
            raise FileNotFoundError(split_root / "images")
        if not (split_root / "labels").is_dir():
            raise FileNotFoundError(split_root / "labels")

    output_root.mkdir(parents=True, exist_ok=False)

    summary_rows: list[dict[str, object]] = []
    total_images = 0
    total_labels = 0
    total_boxes = 0
    total_polygons = 0
    total_duplicates = 0

    for split, split_root in split_sources.items():
        source_images = split_root / "images"
        source_labels = split_root / "labels"
        destination_images = output_root / split / "images"
        destination_labels = output_root / split / "labels"

        destination_images.mkdir(parents=True, exist_ok=True)
        destination_labels.mkdir(parents=True, exist_ok=True)

        image_paths = sorted(
            path for path in source_images.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        split_counts: Counter = Counter()
        split_instances = 0
        split_boxes = 0
        split_polygons = 0
        split_duplicates = 0

        for image_path in image_paths:
            label_path = source_labels / f"{image_path.stem}.txt"
            if not label_path.is_file():
                raise FileNotFoundError(
                    f"Missing label for image: {image_path}"
                )

            shutil.copy2(image_path, destination_images / image_path.name)

            counts, boxes, polygons, duplicates = convert_label(
                label_path,
                destination_labels / label_path.name,
                old_to_new,
            )

            split_counts.update(counts)
            split_instances += sum(counts.values())
            split_boxes += boxes
            split_polygons += polygons
            split_duplicates += duplicates

        orphan_labels = [
            path.name for path in source_labels.glob("*.txt")
            if not any(
                (source_images / f"{path.stem}{extension}").exists()
                for extension in IMAGE_EXTENSIONS
            )
        ]
        if orphan_labels:
            raise RuntimeError(
                f"Found labels without matching images in {split}: "
                f"{orphan_labels[:10]}"
            )

        total_images += len(image_paths)
        total_labels += len(image_paths)
        total_boxes += split_boxes
        total_polygons += split_polygons
        total_duplicates += split_duplicates

        row: dict[str, object] = {
            "split": split,
            "images": len(image_paths),
            "labels": len(image_paths),
            "instances": split_instances,
            "source_boxes": split_boxes,
            "source_polygons": split_polygons,
            "duplicates_removed": split_duplicates,
        }
        for class_id, class_name in enumerate(TARGET_NAMES):
            row[class_name] = split_counts[class_id]
        summary_rows.append(row)

    yaml_path = output_root / "data.yaml"
    with yaml_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {
                "path": str(output_root),
                "train": "train/images",
                "val": "val/images",
                "test": "test/images",
                "nc": 10,
                "names": TARGET_NAMES,
            },
            handle,
            sort_keys=False,
            allow_unicode=True,
        )

    summary_path = output_root / "conversion_summary.csv"
    fieldnames = [
        "split",
        "images",
        "labels",
        "instances",
        "source_boxes",
        "source_polygons",
        "duplicates_removed",
        *TARGET_NAMES,
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\nConversion complete.")
    print(f"Output dataset: {output_root}")
    print(f"YAML: {yaml_path}")
    print(f"Summary: {summary_path}")
    print(f"Total images copied: {total_images}")
    print(f"Total label files written: {total_labels}")
    print(f"Source detection boxes: {total_boxes}")
    print(f"Source polygons converted: {total_polygons}")
    print(f"Duplicates removed: {total_duplicates}")


if __name__ == "__main__":
    main()
