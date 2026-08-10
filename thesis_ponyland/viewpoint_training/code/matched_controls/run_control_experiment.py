from __future__ import annotations

import argparse
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path

from matched_control_lib import (
    DEFAULT_CONTROL_EXPERIMENT_ROOT,
    DEFAULT_FULL_BASELINE_SUMMARY,
    build_control_subset,
    control_dir,
    control_job_by_id,
    control_manifest_path,
    control_metadata_path,
    control_status_path,
    default_training_args,
    ensure_control_experiment_root,
    evaluate_coco,
    install_ultralytics_unique_label_cache,
    load_baseline_summary_row,
    load_or_build_coco_gt_for_images,
    load_yolo,
    predict_yolo_to_coco_json,
    read_json,
    resolve_split_images,
    setup_yolo_environment,
    write_image_list,
    write_json_atomic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one resumable equal-image-count full-M4 control: subset, train, and evaluate.",
    )
    parser.add_argument("--control-id", required=True, help="Matched-control id from the generated manifest.")
    parser.add_argument(
        "--base-data-yaml",
        default=r"C:\DATA\airsim\thesis\captures\S0_20251219_164144\dataset\M4_fixed.yaml",
        help="Base M4 dataset YAML used to build subsets and fixed test assets.",
    )
    parser.add_argument(
        "--experiment-root",
        default=str(DEFAULT_CONTROL_EXPERIMENT_ROOT),
        help="Root directory for matched-control outputs.",
    )
    parser.add_argument(
        "--controls-csv",
        default="",
        help="Optional explicit matched-control manifest path. Defaults to the experiment manifest.",
    )
    parser.add_argument(
        "--stage",
        choices=["subset", "train", "eval", "all", "sanity"],
        default="all",
        help="Subset only, train only, eval only, full pipeline, or lightweight subset sanity only.",
    )
    parser.add_argument("--device", default=None, help="Training device passed to Ultralytics, e.g. '0' or 'cpu'.")
    parser.add_argument("--eval-device", default=None, help="Optional separate device for evaluation predictions.")
    parser.add_argument("--epochs", type=int, default=None, help="Optional training epoch override.")
    parser.add_argument("--batch", type=int, default=None, help="Optional training batch-size override.")
    parser.add_argument("--imgsz", type=int, default=None, help="Optional image-size override used for train/eval.")
    parser.add_argument("--workers", type=int, default=None, help="Optional dataloader worker override.")
    parser.add_argument("--model", default=None, help="Optional model weights for training, defaults to yolov8l.pt.")
    parser.add_argument("--eval-batch", type=int, default=16, help="Batch size for evaluation prediction caching.")
    parser.add_argument("--eval-conf", type=float, default=0.001, help="Confidence threshold for evaluation inference.")
    parser.add_argument("--force-subset", action="store_true", help="Rebuild the subset metadata and list files.")
    parser.add_argument("--force-train", action="store_true", help="Ignore completed training artifacts and retrain.")
    parser.add_argument("--force-eval", action="store_true", help="Ignore cached evaluation predictions and rerun them.")
    parser.add_argument("--resume-training", action="store_true", help="Resume from an existing last.pt if present.")
    parser.add_argument(
        "--baseline-summary-csv",
        default=str(DEFAULT_FULL_BASELINE_SUMMARY),
        help="Optional baseline summary CSV used for gain/loss deltas in the status file.",
    )
    return parser.parse_args()


def resolve_runtime_device(requested_device: str | None) -> str | None:
    if requested_device is None:
        return None
    if requested_device.lower() == "cpu":
        return requested_device

    slurm_step_gpus = os.environ.get("SLURM_STEP_GPUS", "").strip()
    slurm_job_gpus = os.environ.get("SLURM_JOB_GPUS", "").strip()
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()

    if slurm_step_gpus:
        return slurm_step_gpus.split(",")[0]
    if slurm_job_gpus:
        return slurm_job_gpus.split(",")[0]
    if cuda_visible_devices:
        return "0"
    return requested_device


def initial_status(control_job) -> dict[str, object]:
    return {
        "control": {
            "control_id": control_job.control_id,
            "label": control_job.label,
            "source_group": control_job.source_group,
            "source_id": control_job.source_id,
            "source_label": control_job.source_label,
            "train_count": control_job.train_count,
            "val_count": control_job.val_count,
            "seed": control_job.seed,
            "sampling_strategy": control_job.sampling_strategy,
            "reference_metrics": {
                "map50_95": control_job.reference_map50_95,
                "map50": control_job.reference_map50,
                "f1": control_job.reference_f1,
            },
        },
        "subset": {"status": "pending"},
        "training": {"status": "pending"},
        "evaluation": {"status": "pending"},
        "baseline": {},
        "errors": [],
        "updated_at": "",
    }


def load_or_init_status(status_path: Path, args: argparse.Namespace, control_job) -> dict[str, object]:
    if status_path.exists():
        status = read_json(status_path)
    else:
        status = initial_status(control_job)
    baseline_summary = Path(args.baseline_summary_csv).resolve()
    baseline_row = load_baseline_summary_row(baseline_summary)
    if baseline_row is not None:
        status["baseline"] = baseline_row
    return status


def save_status(status_path: Path, status: dict[str, object]) -> None:
    status["updated_at"] = datetime.now(timezone.utc).isoformat()
    write_json_atomic(status_path, status)


def subset_stage(args: argparse.Namespace, status: dict[str, object], control_job) -> dict[str, object]:
    metadata = build_control_subset(
        Path(args.base_data_yaml).resolve(),
        Path(args.experiment_root).resolve(),
        control_job,
        force=args.force_subset,
    )
    status["subset"] = {
        "status": "completed",
        "metadata_path": str(control_metadata_path(Path(args.experiment_root).resolve(), control_job)),
        "data_yaml": metadata["subset_files"]["data_yaml"],
        "split_counts": metadata["split_counts"],
        "missing_labels": metadata["missing_labels"],
    }
    return metadata


def training_stage(args: argparse.Namespace, status: dict[str, object], control_job, project_root: Path) -> Path:
    experiment_root = Path(args.experiment_root).resolve()
    metadata = read_json(control_metadata_path(experiment_root, control_job))
    job_dir = control_dir(experiment_root, control_job)
    run_dir = job_dir / "train"
    best_path = run_dir / "weights" / "best.pt"
    last_path = run_dir / "weights" / "last.pt"

    if best_path.exists() and not args.force_train:
        status["training"] = {
            "status": "completed",
            "run_dir": str(run_dir.resolve()),
            "model_path": str(best_path.resolve()),
            "resumed": False,
            "settings": status.get("training", {}).get("settings", {}),
        }
        return best_path

    setup_yolo_environment(project_root)
    install_ultralytics_unique_label_cache(job_dir / "ultralytics_label_cache")
    YOLO = load_yolo()
    base_defaults = default_training_args()

    try:
        if args.resume_training and last_path.exists() and not args.force_train:
            model = YOLO(str(last_path))
            model.train(resume=True)
            resumed = True
        else:
            train_kwargs = dict(base_defaults)
            base_model = str(args.model or train_kwargs.pop("model"))
            train_kwargs["data"] = str(metadata["subset_files"]["data_yaml"])
            train_kwargs["project"] = str(job_dir.resolve())
            train_kwargs["name"] = "train"
            train_kwargs["exist_ok"] = True
            if args.device is not None:
                train_kwargs["device"] = args.device
            if args.epochs is not None:
                train_kwargs["epochs"] = args.epochs
            if args.batch is not None:
                train_kwargs["batch"] = args.batch
            if args.imgsz is not None:
                train_kwargs["imgsz"] = args.imgsz
            if args.workers is not None:
                train_kwargs["workers"] = args.workers

            model = YOLO(base_model)
            model.train(**train_kwargs)
            resumed = False
        model_path = best_path if best_path.exists() else last_path
        status["training"] = {
            "status": "completed",
            "run_dir": str(run_dir.resolve()),
            "model_path": str(model_path.resolve()),
            "resumed": resumed,
            "settings": {
                **base_defaults,
                "model": str(args.model or base_defaults["model"]),
                **({} if args.device is None else {"device": args.device}),
                **({} if args.epochs is None else {"epochs": args.epochs}),
                **({} if args.batch is None else {"batch": args.batch}),
                **({} if args.imgsz is None else {"imgsz": args.imgsz}),
                **({} if args.workers is None else {"workers": args.workers}),
            },
        }
        return model_path
    except Exception as exc:  # pragma: no cover - runtime protection
        status["training"] = {
            "status": "failed",
            "run_dir": str(run_dir.resolve()),
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        raise


def ensure_full_test_assets(args: argparse.Namespace, metadata: dict[str, object], experiment_root: Path):
    shared_dir = experiment_root / "shared"
    shared_dir.mkdir(parents=True, exist_ok=True)
    full_test_list = shared_dir / "full_test.txt"
    full_test_gt = shared_dir / "full_test_gt.json"
    full_test_images = resolve_split_images(Path(args.base_data_yaml).resolve(), "test")
    if not full_test_list.exists():
        write_image_list(full_test_list, full_test_images)
    ordered_paths, image_id_map = load_or_build_coco_gt_for_images(full_test_images, metadata["class_names"], full_test_gt)
    return ordered_paths, full_test_gt, image_id_map


def evaluation_stage(args: argparse.Namespace, status: dict[str, object], control_job, model_path: Path, project_root: Path) -> None:
    experiment_root = Path(args.experiment_root).resolve()
    metadata = read_json(control_metadata_path(experiment_root, control_job))
    eval_dir = control_dir(experiment_root, control_job) / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    eval_device = args.eval_device if args.eval_device is not None else args.device
    base_defaults = default_training_args()
    eval_imgsz = args.imgsz if args.imgsz is not None else int(base_defaults["imgsz"])

    full_test_images, full_gt_json, full_image_id_map = ensure_full_test_assets(args, metadata, experiment_root)
    pred_json = eval_dir / "full_test_predictions.json"
    metrics_path = eval_dir / "full_test_metrics.json"
    if args.force_eval or not pred_json.exists():
        predict_yolo_to_coco_json(
            model_path=model_path,
            image_paths=full_test_images,
            image_id_map=full_image_id_map,
            out_json=pred_json,
            imgsz=eval_imgsz,
            conf=args.eval_conf,
            batch=args.eval_batch,
            device=eval_device,
            project_root=project_root,
        )
    metrics = evaluate_coco(full_gt_json, pred_json)
    metrics["num_test_images"] = len(full_test_images)

    reference = status["control"]["reference_metrics"]
    baseline = status.get("baseline", {})
    metrics["delta_map50_95_vs_reference"] = float(metrics["map50_95"]) - float(reference["map50_95"])
    metrics["delta_map50_vs_reference"] = float(metrics["map50"]) - float(reference["map50"])
    metrics["delta_f1_vs_reference"] = float(metrics["f1"]) - float(reference["f1"])
    if baseline:
        metrics["delta_map50_95_vs_full_m4"] = float(metrics["map50_95"]) - float(baseline["map50_95"])
        metrics["delta_map50_vs_full_m4"] = float(metrics["map50"]) - float(baseline["map50"])
        metrics["delta_f1_vs_full_m4"] = float(metrics["f1"]) - float(baseline["f1"])

    write_json_atomic(metrics_path, metrics)
    status["evaluation"] = {
        "status": "completed",
        "prediction_json": str(pred_json.resolve()),
        "metrics_json": str(metrics_path.resolve()),
        "metrics": metrics,
    }


def main() -> None:
    args = parse_args()
    args.device = resolve_runtime_device(args.device)
    args.eval_device = resolve_runtime_device(args.eval_device) if args.eval_device is not None else args.device
    print(
        "Resolved runtime devices:",
        f"train={args.device}",
        f"eval={args.eval_device}",
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}",
        f"SLURM_STEP_GPUS={os.environ.get('SLURM_STEP_GPUS', 'unset')}",
        f"SLURM_JOB_GPUS={os.environ.get('SLURM_JOB_GPUS', 'unset')}",
    )

    project_root = Path(__file__).resolve().parents[1]
    experiment_root = Path(args.experiment_root).resolve()
    ensure_control_experiment_root(experiment_root)
    controls_csv = Path(args.controls_csv).resolve() if args.controls_csv else control_manifest_path(experiment_root)
    control_job = control_job_by_id(controls_csv, args.control_id)
    status_path = control_status_path(experiment_root, control_job)
    status = load_or_init_status(status_path, args, control_job)
    save_status(status_path, status)

    try:
        if args.stage in {"subset", "all", "sanity"}:
            subset_stage(args, status, control_job)
            save_status(status_path, status)

        if args.stage == "subset":
            print(f"Subset build completed for {control_job.control_id}: {status_path}")
            return

        if args.stage == "sanity":
            print(f"Subset sanity completed for {control_job.control_id}: {status_path}")
            return

        if args.stage in {"train", "all"}:
            model_path = training_stage(args, status, control_job, project_root)
            save_status(status_path, status)
        else:
            training_info = status.get("training", {})
            model_path_raw = training_info.get("model_path")
            if not model_path_raw:
                raise FileNotFoundError("No trained model path is available. Run the training stage first.")
            model_path = Path(str(model_path_raw))

        if args.stage in {"eval", "all"}:
            evaluation_stage(args, status, control_job, model_path, project_root)
            save_status(status_path, status)

        print(f"Completed {args.stage} for {control_job.control_id}. Status: {status_path}")
    except Exception as exc:  # pragma: no cover - runtime protection
        status.setdefault("errors", []).append({"message": str(exc), "traceback": traceback.format_exc()})
        save_status(status_path, status)
        raise


if __name__ == "__main__":
    main()
