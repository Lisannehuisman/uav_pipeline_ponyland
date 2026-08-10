from __future__ import annotations

import argparse
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path

from pair_experiment_lib import (
    DEFAULT_FULL_BASELINE_SUMMARY,
    DEFAULT_PROTOCOL_RECOMMENDATION,
    DEFAULT_TRAINING_ARGS,
    build_pair_subset,
    ensure_experiment_root,
    load_baseline_summary_row,
    load_or_build_coco_gt_for_images,
    pair_dir,
    pair_job_by_id,
    pair_manifest_path,
    pair_metadata_path,
    pair_status_path,
    predict_yolo_to_coco_json,
    read_image_list,
    read_json,
    resolve_split_images,
    install_ultralytics_unique_label_cache,
    setup_yolo_environment,
    write_image_list,
    write_json_atomic,
    evaluate_coco,
    load_yolo,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one resumable M4 pair-subset experiment job: subset, train, and evaluate.",
    )
    parser.add_argument("--pair-id", required=True, help="Pair id from the generated pair manifest.")
    parser.add_argument(
        "--base-data-yaml",
        default=r"C:\DATA\airsim\thesis\captures\S0_20251219_164144\dataset\M4_fixed.yaml",
        help="Base M4 dataset YAML used to build subsets and fixed test assets.",
    )
    parser.add_argument(
        "--experiment-root",
        default="outputs/m4_pair_subset_experiment",
        help="Root directory for pair-subset outputs.",
    )
    parser.add_argument(
        "--pairs-csv",
        default="",
        help="Optional explicit pair manifest path. Defaults to the experiment manifest.",
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
    parser.add_argument("--skip-option-b", action="store_true", help="Skip the same-pair test evaluation.")
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
        # When the scheduler masks visibility, the assigned GPU is exposed as logical device 0.
        return "0"
    return requested_device


def initial_status(args: argparse.Namespace, pair_job) -> dict[str, object]:
    return {
        "pair": {
            "pair_id": pair_job.pair_id,
            "pair_index": pair_job.pair_index,
            "viewpoint_1": pair_job.viewpoint_1,
            "viewpoint_2": pair_job.viewpoint_2,
            "pilot_rank": pair_job.pilot_rank,
            "pilot_name": pair_job.pilot_name,
            "pilot_note": pair_job.pilot_note,
        },
        "protocol": {
            "recommended_default": DEFAULT_PROTOCOL_RECOMMENDATION,
            "option_a": "Evaluate the pair-trained model on the full fixed M4 test split across all 72 viewpoints.",
            "option_b": "Evaluate the pair-trained model only on the same two viewpoints in the fixed M4 test split.",
        },
        "subset": {"status": "pending"},
        "training": {"status": "pending"},
        "evaluations": {
            "option_a_full_test": {"status": "pending"},
            "option_b_pair_test": {"status": "pending"},
        },
        "baseline": {},
        "errors": [],
        "updated_at": "",
    }


def load_or_init_status(status_path: Path, args: argparse.Namespace, pair_job) -> dict[str, object]:
    if status_path.exists():
        status = read_json(status_path)
    else:
        status = initial_status(args, pair_job)
    baseline_summary = Path(args.baseline_summary_csv).resolve()
    baseline_row = load_baseline_summary_row(baseline_summary)
    if baseline_row is not None:
        status["baseline"] = baseline_row
    return status


def save_status(status_path: Path, status: dict[str, object]) -> None:
    status["updated_at"] = datetime.now(timezone.utc).isoformat()
    write_json_atomic(status_path, status)


def subset_stage(args: argparse.Namespace, status: dict[str, object], pair_job) -> dict[str, object]:
    metadata = build_pair_subset(
        Path(args.base_data_yaml).resolve(),
        Path(args.experiment_root).resolve(),
        pair_job,
        force=args.force_subset,
    )
    status["subset"] = {
        "status": "completed",
        "metadata_path": str(pair_metadata_path(Path(args.experiment_root).resolve(), pair_job)),
        "data_yaml": metadata["subset_files"]["data_yaml"],
        "split_counts": metadata["split_counts"],
        "missing_labels": metadata["missing_labels"],
    }
    return metadata


def training_stage(args: argparse.Namespace, status: dict[str, object], pair_job, project_root: Path) -> Path:
    experiment_root = Path(args.experiment_root).resolve()
    metadata = read_json(pair_metadata_path(experiment_root, pair_job))
    job_dir = pair_dir(experiment_root, pair_job)
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

    try:
        if args.resume_training and last_path.exists() and not args.force_train:
            model = YOLO(str(last_path))
            model.train(resume=True)
            resumed = True
        else:
            train_kwargs = dict(DEFAULT_TRAINING_ARGS)
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
                **DEFAULT_TRAINING_ARGS,
                "model": str(args.model or DEFAULT_TRAINING_ARGS["model"]),
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


def ensure_full_test_assets(args: argparse.Namespace, metadata: dict[str, object], experiment_root: Path) -> tuple[list[Path], Path, dict[str, int]]:
    shared_dir = experiment_root / "shared"
    shared_dir.mkdir(parents=True, exist_ok=True)
    full_test_list = shared_dir / "full_test.txt"
    full_test_gt = shared_dir / "full_test_gt.json"
    full_test_images = resolve_split_images(Path(args.base_data_yaml).resolve(), "test")
    if not full_test_list.exists():
        write_image_list(full_test_list, full_test_images)
    ordered_paths, image_id_map = load_or_build_coco_gt_for_images(full_test_images, metadata["class_names"], full_test_gt)
    return ordered_paths, full_test_gt, image_id_map


def evaluation_stage(args: argparse.Namespace, status: dict[str, object], pair_job, model_path: Path, project_root: Path) -> None:
    experiment_root = Path(args.experiment_root).resolve()
    metadata = read_json(pair_metadata_path(experiment_root, pair_job))
    eval_dir = pair_dir(experiment_root, pair_job) / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    eval_device = args.eval_device if args.eval_device is not None else args.device
    eval_imgsz = args.imgsz if args.imgsz is not None else int(DEFAULT_TRAINING_ARGS["imgsz"])

    full_test_images, full_gt_json, full_image_id_map = ensure_full_test_assets(args, metadata, experiment_root)
    option_a_pred = eval_dir / "option_a_full_test_predictions.json"
    option_a_metrics_path = eval_dir / "option_a_full_test_metrics.json"
    if args.force_eval or not option_a_pred.exists():
        predict_yolo_to_coco_json(
            model_path=model_path,
            image_paths=full_test_images,
            image_id_map=full_image_id_map,
            out_json=option_a_pred,
            imgsz=eval_imgsz,
            conf=args.eval_conf,
            batch=args.eval_batch,
            device=eval_device,
            project_root=project_root,
        )
    option_a_metrics = evaluate_coco(full_gt_json, option_a_pred)
    option_a_metrics["num_test_images"] = len(full_test_images)
    write_json_atomic(option_a_metrics_path, option_a_metrics)
    status["evaluations"]["option_a_full_test"] = {
        "status": "completed",
        "prediction_json": str(option_a_pred.resolve()),
        "metrics_json": str(option_a_metrics_path.resolve()),
        "metrics": option_a_metrics,
    }

    if args.skip_option_b:
        status["evaluations"]["option_b_pair_test"] = {"status": "skipped"}
        return

    pair_test_images = read_image_list(Path(metadata["subset_files"]["test_pair_list"]))
    option_b_gt = eval_dir / "option_b_pair_test_gt.json"
    pair_ordered_images, pair_image_id_map = load_or_build_coco_gt_for_images(
        pair_test_images,
        metadata["class_names"],
        option_b_gt,
    )
    option_b_pred = eval_dir / "option_b_pair_test_predictions.json"
    option_b_metrics_path = eval_dir / "option_b_pair_test_metrics.json"
    if args.force_eval or not option_b_pred.exists():
        predict_yolo_to_coco_json(
            model_path=model_path,
            image_paths=pair_ordered_images,
            image_id_map=pair_image_id_map,
            out_json=option_b_pred,
            imgsz=eval_imgsz,
            conf=args.eval_conf,
            batch=args.eval_batch,
            device=eval_device,
            project_root=project_root,
        )
    option_b_metrics = evaluate_coco(option_b_gt, option_b_pred)
    option_b_metrics["num_test_images"] = len(pair_ordered_images)
    write_json_atomic(option_b_metrics_path, option_b_metrics)
    status["evaluations"]["option_b_pair_test"] = {
        "status": "completed",
        "prediction_json": str(option_b_pred.resolve()),
        "metrics_json": str(option_b_metrics_path.resolve()),
        "metrics": option_b_metrics,
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
    ensure_experiment_root(experiment_root)
    pairs_csv = Path(args.pairs_csv).resolve() if args.pairs_csv else pair_manifest_path(experiment_root)
    pair_job = pair_job_by_id(pairs_csv, args.pair_id)
    status_path = pair_status_path(experiment_root, pair_job)
    status = load_or_init_status(status_path, args, pair_job)
    save_status(status_path, status)

    try:
        if args.stage in {"subset", "all", "sanity"}:
            subset_stage(args, status, pair_job)
            save_status(status_path, status)

        if args.stage == "sanity":
            print(f"Subset sanity completed for {pair_job.pair_id}: {status_path}")
            return

        if args.stage in {"train", "all"}:
            model_path = training_stage(args, status, pair_job, project_root)
            save_status(status_path, status)
        else:
            training_info = status.get("training", {})
            model_path_raw = training_info.get("model_path")
            if not model_path_raw:
                raise FileNotFoundError("No trained model path is available. Run the training stage first.")
            model_path = Path(str(model_path_raw))

        if args.stage in {"eval", "all"}:
            evaluation_stage(args, status, pair_job, model_path, project_root)
            save_status(status_path, status)

        print(f"Completed {args.stage} for {pair_job.pair_id}. Status: {status_path}")
    except Exception as exc:  # pragma: no cover - runtime protection
        status.setdefault("errors", []).append({"message": str(exc), "traceback": traceback.format_exc()})
        save_status(status_path, status)
        raise


if __name__ == "__main__":
    main()
