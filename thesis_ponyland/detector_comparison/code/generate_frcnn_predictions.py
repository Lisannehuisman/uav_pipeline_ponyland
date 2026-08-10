from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.ops import FrozenBatchNorm2d

from comparison_config import DEFAULT_OUTPUT_DIR, MODEL_RUNS, REGIME_DATA_YAMLS, REGIME_ORDER


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
PIXEL_MEAN = [103.53, 116.28, 123.675]
PIXEL_STD = [1.0, 1.0, 1.0]
NUM_CLASSES = 11


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate COCO prediction JSONs for Faster R-CNN checkpoints using torchvision inference.",
    )
    parser.add_argument("--split", choices=["test", "val"], default="test", help="Dataset split to run.")
    parser.add_argument(
        "--regimes",
        nargs="+",
        default=REGIME_ORDER,
        choices=REGIME_ORDER,
        help="Regimes to process.",
    )
    parser.add_argument("--device", default="cpu", help="Torch device, for example 'cpu'.")
    parser.add_argument("--score-thresh", type=float, default=0.05, help="Score threshold for kept detections.")
    parser.add_argument("--topk", type=int, default=100, help="Maximum detections per image.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR / "standardized_test_eval" / "predictions"),
        help="Directory for generated COCO JSONs.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing jsonl cache if present.",
    )
    return parser.parse_args()


def resolve_frcnn_run_dir(run_dir: Path) -> Path:
    if (run_dir / "model_final.pth").exists():
        return run_dir
    nested_candidates = list(run_dir.glob("*/model_final.pth"))
    if nested_candidates:
        return nested_candidates[0].parent
    return run_dir


def resolve_dataset_root(data_yaml: Path, data_dict: dict) -> Path:
    configured_root = data_dict.get("path")
    if configured_root:
        root = Path(configured_root)
        if not root.is_absolute():
            root = (data_yaml.parent / root).resolve()
        return root
    return data_yaml.parent.resolve()


def resolve_split_images(data_yaml: Path, split: str) -> tuple[dict, list[Path]]:
    with data_yaml.open("r", encoding="utf-8") as handle:
        data_dict = yaml.safe_load(handle)

    split_value = data_dict.get(split)
    if split_value is None:
        raise ValueError(f"Split '{split}' was not found in {data_yaml}.")

    root = resolve_dataset_root(data_yaml, data_dict)
    candidates = [Path(split_value)] if isinstance(split_value, str) else [Path(item) for item in split_value]
    images: list[Path] = []

    for candidate in candidates:
        resolved = candidate if candidate.is_absolute() else (root / candidate).resolve()
        if resolved.is_dir():
            for path in resolved.rglob("*"):
                if path.suffix.lower() in IMAGE_EXTENSIONS:
                    images.append(path)
        elif resolved.is_file() and resolved.suffix.lower() == ".txt":
            with resolved.open("r", encoding="utf-8") as handle:
                for line in handle:
                    raw = line.strip()
                    if not raw:
                        continue
                    listed = Path(raw)
                    if not listed.is_absolute():
                        listed = (resolved.parent / listed).resolve()
                    if listed.suffix.lower() in IMAGE_EXTENSIONS:
                        images.append(listed)
        elif resolved.is_file() and resolved.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(resolved)
        else:
            raise FileNotFoundError(f"Could not resolve image path '{candidate}' from split '{split}'.")

    return data_dict, sorted(images)


def resolve_official_coco_gt(data_yaml: Path, split: str) -> Path | None:
    if split != "val":
        return None

    with data_yaml.open("r", encoding="utf-8") as handle:
        data_dict = yaml.safe_load(handle)

    dataset_root = resolve_dataset_root(data_yaml, data_dict)
    regime_name = data_yaml.stem
    candidates = [
        dataset_root / "coco_annotations" / f"coco_instances_{split}_{regime_name}.json",
        dataset_root / "annotations" / f"instances_{split}_{regime_name}.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def image_id_mapping(data_yaml: Path, split: str) -> tuple[list[Path], dict[str, int]]:
    official_gt = resolve_official_coco_gt(data_yaml, split)
    if official_gt is None:
        _, image_paths = resolve_split_images(data_yaml, split)
        image_id_map = {str(path): image_id for image_id, path in enumerate(image_paths, start=1)}
        return image_paths, image_id_map

    gt = json.loads(official_gt.read_text(encoding="utf-8"))
    _, available_images = resolve_split_images(data_yaml, split)
    images_by_name = {image_path.name: image_path for image_path in available_images}

    ordered_paths: list[Path] = []
    image_id_map: dict[str, int] = {}
    for image_info in gt["images"]:
        image_path = images_by_name[str(image_info["file_name"])]
        ordered_paths.append(image_path)
        image_id_map[str(image_path)] = int(image_info["id"])
    return ordered_paths, image_id_map


def convert_checkpoint_state_dict(checkpoint_model: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    converted: dict[str, torch.Tensor] = {}

    for key, value in checkpoint_model.items():
        mapped_key: str
        pad_background = False

        if key.startswith("backbone.bottom_up.stem.conv1."):
            tail = key[len("backbone.bottom_up.stem.conv1.") :]
            if tail == "weight":
                mapped_key = "backbone.body.conv1.weight"
            else:
                mapped_key = "backbone.body.bn1." + tail.replace("norm.", "")
        else:
            match = re.match(r"backbone\.bottom_up\.res([2-5])\.(\d+)\.(.+)", key)
            if match:
                stage = int(match.group(1)) - 1
                block = match.group(2)
                tail = match.group(3)
                base = f"backbone.body.layer{stage}.{block}."
                if tail == "shortcut.weight":
                    mapped_key = base + "downsample.0.weight"
                elif tail.startswith("shortcut.norm."):
                    mapped_key = base + "downsample.1." + tail[len("shortcut.norm.") :]
                elif ".norm." in tail:
                    conv_name, rest = tail.split(".norm.", 1)
                    mapped_key = base + conv_name.replace("conv", "bn") + "." + rest
                else:
                    mapped_key = base + tail
            else:
                match = re.match(r"backbone\.fpn_lateral([2-5])\.(weight|bias)", key)
                if match:
                    mapped_key = f"backbone.fpn.inner_blocks.{int(match.group(1)) - 2}.0.{match.group(2)}"
                else:
                    match = re.match(r"backbone\.fpn_output([2-5])\.(weight|bias)", key)
                    if match:
                        mapped_key = f"backbone.fpn.layer_blocks.{int(match.group(1)) - 2}.0.{match.group(2)}"
                    elif key.startswith("proposal_generator.rpn_head.conv."):
                        mapped_key = "rpn.head.conv.0.0." + key.split(".")[-1]
                    elif key.startswith("proposal_generator.rpn_head.objectness_logits."):
                        mapped_key = "rpn.head.cls_logits." + key.split(".")[-1]
                    elif key.startswith("proposal_generator.rpn_head.anchor_deltas."):
                        mapped_key = "rpn.head.bbox_pred." + key.split(".")[-1]
                    elif key.startswith("roi_heads.box_head.fc1."):
                        mapped_key = "roi_heads.box_head.fc6." + key.split(".")[-1]
                    elif key.startswith("roi_heads.box_head.fc2."):
                        mapped_key = "roi_heads.box_head.fc7." + key.split(".")[-1]
                    elif key.startswith("roi_heads.box_predictor.cls_score."):
                        mapped_key = key
                        # Detectron2 stores the background logit in the last row, while
                        # torchvision expects the background class first.
                        value = torch.cat([value[-1:].clone(), value[:-1].clone()], dim=0)
                    elif key.startswith("roi_heads.box_predictor.bbox_pred."):
                        mapped_key = "roi_heads.box_predictor.bbox_pred." + key.split(".")[-1]
                        pad_background = True
                    else:
                        raise KeyError(f"Unrecognized checkpoint key: {key}")

        if pad_background:
            if mapped_key.endswith("weight"):
                padded = torch.zeros((44, value.shape[1]), dtype=value.dtype)
                padded[4:] = value
            else:
                padded = torch.zeros((44,), dtype=value.dtype)
                padded[4:] = value
            converted[mapped_key] = padded
        else:
            converted[mapped_key] = value

    return converted


def build_model(weights_path: Path, score_thresh: float, topk: int, device: str) -> FasterRCNN:
    checkpoint = torch.load(weights_path, map_location="cpu")
    state_dict = convert_checkpoint_state_dict(checkpoint["model"])

    backbone = resnet_fpn_backbone(
        backbone_name="resnet50",
        weights=None,
        norm_layer=FrozenBatchNorm2d,
    )
    model = FasterRCNN(
        backbone,
        num_classes=NUM_CLASSES,
        min_size=800,
        max_size=1333,
        image_mean=PIXEL_MEAN,
        image_std=PIXEL_STD,
        box_score_thresh=score_thresh,
        box_detections_per_img=topk,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_image_tensor(image_path: Path) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    array = np.array(image, dtype=np.float32)
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    # Detectron2 checkpoints expect BGR channel order with 0-255 pixel range.
    return tensor[[2, 1, 0], :, :]


def read_existing_jsonl(jsonl_path: Path) -> tuple[list[dict], set[int]]:
    if not jsonl_path.exists():
        return [], set()

    rows: list[dict] = []
    completed_image_ids: set[int] = set()
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows.append(row)
            completed_image_ids.add(int(row["image_id"]))
    return rows, completed_image_ids


def write_final_json(predictions: list[dict], output_json: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(predictions), encoding="utf-8")


def infer_regime(
    regime: str,
    split: str,
    output_dir: Path,
    device: str,
    score_thresh: float,
    topk: int,
    resume: bool,
) -> Path:
    run_dir = resolve_frcnn_run_dir(Path(MODEL_RUNS["Faster R-CNN"][regime]))
    weights_path = run_dir / "model_final.pth"
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing Faster R-CNN weights for {regime}: {weights_path}")

    data_yaml = Path(REGIME_DATA_YAMLS[regime]).resolve()
    image_paths, image_id_map = image_id_mapping(data_yaml, split=split)

    output_json = output_dir / f"Faster_R-CNN_{regime}_predictions.json"
    output_jsonl = output_dir / f"Faster_R-CNN_{regime}_predictions.jsonl"
    predictions, completed_image_ids = read_existing_jsonl(output_jsonl) if resume else ([], set())

    model = build_model(weights_path, score_thresh=score_thresh, topk=topk, device=device)

    pending_images = [image_path for image_path in image_paths if image_id_map[str(image_path)] not in completed_image_ids]
    total_images = len(image_paths)
    print(
        f"{regime}: {len(completed_image_ids)}/{total_images} images already cached. {len(pending_images)} to go.",
        flush=True,
    )

    if pending_images:
        output_dir.mkdir(parents=True, exist_ok=True)
        with output_jsonl.open("a", encoding="utf-8") as cache_handle:
            with torch.inference_mode():
                for index, image_path in enumerate(pending_images, start=1):
                    image_tensor = load_image_tensor(image_path).to(device)
                    result = model([image_tensor])[0]
                    image_id = image_id_map[str(image_path)]

                    boxes = result["boxes"].detach().cpu().numpy()
                    labels = result["labels"].detach().cpu().numpy()
                    scores = result["scores"].detach().cpu().numpy()

                    image_predictions = []
                    for box, label, score in zip(boxes, labels, scores):
                        x1, y1, x2, y2 = box.tolist()
                        image_predictions.append(
                            {
                                "image_id": int(image_id),
                                "category_id": int(label),
                                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                                "score": float(score),
                            }
                        )

                    for row in image_predictions:
                        cache_handle.write(json.dumps(row) + "\n")
                    predictions.extend(image_predictions)

                    progress = len(completed_image_ids) + index
                    if progress % 25 == 0 or progress == total_images:
                        print(f"{regime}: processed {progress}/{total_images} images.", flush=True)

    write_final_json(predictions, output_json)
    print(f"Saved {regime} predictions to: {output_json}", flush=True)
    return output_json


def main() -> None:
    args = parse_args()
    device = args.device
    if device != "cpu" and not torch.cuda.is_available():
        raise SystemExit(f"Requested device '{device}', but CUDA is not available.")

    output_dir = Path(args.output_dir).resolve()
    for regime in args.regimes:
        infer_regime(
            regime=regime,
            split=args.split,
            output_dir=output_dir,
            device=device,
            score_thresh=args.score_thresh,
            topk=args.topk,
            resume=args.resume,
        )


if __name__ == "__main__":
    main()
