from __future__ import annotations

import argparse
import shlex
from pathlib import Path

from pair_experiment_lib import ensure_experiment_root, load_pair_jobs, pair_manifest_path


def shell_value_or_quote(value: str) -> str:
    return value if "${" in value else shlex.quote(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate generic bash or Slurm launch scripts for the M4 pair-subset sweep.",
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
        "--mode",
        choices=["pilot", "full"],
        default="pilot",
        help="Emit launchers for the pilot subset or the full pair sweep.",
    )
    parser.add_argument(
        "--launcher",
        choices=["bash", "slurm"],
        default="bash",
        help="Type of launcher script to emit.",
    )
    parser.add_argument(
        "--python-executable",
        default=".venv/bin/python",
        help="Python executable to use in the generated launch script.",
    )
    parser.add_argument(
        "--job-name",
        default="m4-pair-sweep",
        help="Slurm job name written into generated array launchers.",
    )
    parser.add_argument(
        "--workspace-root",
        default=str(Path.cwd()),
        help="Workspace root that the generated launcher should `cd` into before running commands.",
    )
    parser.add_argument(
        "--base-data-yaml",
        default="/vol/tensusers6/lisannehuisman/yamls/M4_yolov8l.yaml",
        help="Dataset YAML to use on the target execution environment.",
    )
    parser.add_argument("--device", default="0", help="Training device for the generated commands.")
    parser.add_argument("--eval-device", default="", help="Optional evaluation device override.")
    parser.add_argument("--model", default="", help="Optional local weights path to pass to the worker.")
    parser.add_argument("--epochs", type=int, default=0, help="Optional epoch override written into the launcher.")
    parser.add_argument("--batch", type=int, default=0, help="Optional training batch-size override.")
    parser.add_argument("--imgsz", type=int, default=0, help="Optional image-size override.")
    parser.add_argument("--workers", type=int, default=0, help="Optional worker override.")
    parser.add_argument("--eval-batch", type=int, default=16, help="Evaluation batch size for the worker.")
    parser.add_argument("--resume-training", action="store_true", help="Include --resume-training in the launchers.")
    parser.add_argument("--skip-option-b", action="store_true", help="Skip Option B in the launchers.")
    parser.add_argument("--max-parallel", type=int, default=16, help="Maximum concurrent tasks for the Slurm array.")
    parser.add_argument("--slurm-partition", default="", help="Optional Slurm partition to request.")
    parser.add_argument("--slurm-account", default="", help="Optional Slurm account to request.")
    parser.add_argument("--slurm-time", default="", help="Optional Slurm wall-time request, e.g. 04:00:00.")
    parser.add_argument("--slurm-mem", default="", help="Optional Slurm memory request, e.g. 24G.")
    parser.add_argument(
        "--slurm-cpus-per-task",
        type=int,
        default=0,
        help="Optional Slurm CPUs-per-task request.",
    )
    parser.add_argument("--slurm-gres", default="", help="Optional Slurm GRES request, e.g. gpu:1.")
    return parser.parse_args()


def selected_jobs(args: argparse.Namespace, pairs_csv: Path):
    jobs = load_pair_jobs(pairs_csv)
    if args.mode == "pilot":
        jobs = [job for job in jobs if job.pilot_rank is not None]
        jobs = sorted(jobs, key=lambda job: job.pilot_rank or 9999)
    return jobs


def build_worker_command(args: argparse.Namespace, pair_id: str, device_value: str | None = None) -> str:
    pair_value = shell_value_or_quote(pair_id)
    resolved_device = device_value if device_value is not None else args.device
    experiment_root = Path(args.experiment_root).as_posix()
    command = [
        shlex.quote(args.python_executable),
        "m4_pair_subset_experiment/run_pair_experiment.py",
        "--pair-id",
        pair_value,
        "--base-data-yaml",
        shlex.quote(args.base_data_yaml),
        "--experiment-root",
        shlex.quote(experiment_root),
        "--stage",
        "all",
        "--device",
        shell_value_or_quote(resolved_device),
        "--eval-batch",
        str(args.eval_batch),
    ]
    if args.eval_device:
        command.extend(["--eval-device", shell_value_or_quote(args.eval_device)])
    if args.model:
        command.extend(["--model", shlex.quote(args.model)])
    if args.epochs > 0:
        command.extend(["--epochs", str(args.epochs)])
    if args.batch > 0:
        command.extend(["--batch", str(args.batch)])
    if args.imgsz > 0:
        command.extend(["--imgsz", str(args.imgsz)])
    if args.workers > 0:
        command.extend(["--workers", str(args.workers)])
    if args.resume_training:
        command.append("--resume-training")
    if args.skip_option_b:
        command.append("--skip-option-b")
    return " ".join(command)


def build_shell_preamble(args: argparse.Namespace) -> list[str]:
    workspace_root = Path(args.workspace_root).as_posix()
    experiment_root = Path(args.experiment_root).as_posix()
    yolo_config_dir = (Path(args.workspace_root) / "Ultralytics").as_posix()
    preamble = [
        "set -euo pipefail",
        f"cd {shlex.quote(workspace_root)}",
        f"mkdir -p {shlex.quote((Path(experiment_root) / 'logs').as_posix())}",
        "export PYTHONUNBUFFERED=1",
        f'export YOLO_CONFIG_DIR="${{YOLO_CONFIG_DIR:-{yolo_config_dir}}}"',
        'mkdir -p "$YOLO_CONFIG_DIR"',
    ]
    if args.launcher == "slurm":
        default_device = shlex.quote(args.device)
        preamble.extend(
            [
                'M4_DEVICE="${SLURM_STEP_GPUS:-}"',
                'if [ -z "$M4_DEVICE" ] && [ -n "${SLURM_JOB_GPUS:-}" ]; then M4_DEVICE="${SLURM_JOB_GPUS%%,*}"; fi',
                'if [ -n "$M4_DEVICE" ]; then M4_DEVICE="${M4_DEVICE%%,*}"; fi',
                'if [ -z "$M4_DEVICE" ] && [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then M4_DEVICE="0"; fi',
                f'if [ -z "$M4_DEVICE" ]; then M4_DEVICE={default_device}; fi',
                'echo "Resolved M4 device: $M4_DEVICE (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}, SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-unset}, SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset})"',
            ]
        )
    return preamble


def write_bash_launcher(args: argparse.Namespace, pair_ids: list[str], output_path: Path) -> None:
    lines = ["#!/usr/bin/env bash", *build_shell_preamble(args), "", "PAIR_IDS=("]
    for pair_id in pair_ids:
        lines.append(f"  {shlex.quote(pair_id)}")
    lines.extend([")", "", 'for PAIR_ID in "${PAIR_IDS[@]}"; do'])
    lines.append(f"  {build_worker_command(args, '${PAIR_ID}')}")
    lines.extend(
        [
            "done",
            "",
            f"{shlex.quote(args.python_executable)} m4_pair_subset_experiment/aggregate_pair_results.py --experiment-root {shlex.quote(Path(args.experiment_root).as_posix())}",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_slurm_launcher(args: argparse.Namespace, pair_ids: list[str], output_path: Path) -> None:
    array_limit = max(0, len(pair_ids) - 1)
    log_dir = Path(args.experiment_root).as_posix() + "/logs"
    lines = [
        "#!/usr/bin/env bash",
        f"#SBATCH --job-name={args.job_name}",
        f"#SBATCH --array=0-{array_limit}%{args.max_parallel}",
        f"#SBATCH --output={log_dir}/m4_pair_%A_%a.out",
        f"#SBATCH --error={log_dir}/m4_pair_%A_%a.err",
        "#SBATCH --ntasks=1",
    ]
    if args.slurm_partition:
        lines.append(f"#SBATCH --partition={args.slurm_partition}")
    if args.slurm_account:
        lines.append(f"#SBATCH --account={args.slurm_account}")
    if args.slurm_time:
        lines.append(f"#SBATCH --time={args.slurm_time}")
    if args.slurm_mem:
        lines.append(f"#SBATCH --mem={args.slurm_mem}")
    if args.slurm_cpus_per_task > 0:
        lines.append(f"#SBATCH --cpus-per-task={args.slurm_cpus_per_task}")
    if args.slurm_gres:
        lines.append(f"#SBATCH --gres={args.slurm_gres}")
    lines.extend(["", *build_shell_preamble(args), "", "PAIR_IDS=("])
    for pair_id in pair_ids:
        lines.append(f"  {shlex.quote(pair_id)}")
    lines.extend(
        [
            ")",
            'PAIR_ID="${PAIR_IDS[$SLURM_ARRAY_TASK_ID]}"',
            build_worker_command(args, '${PAIR_ID}', device_value='${M4_DEVICE}'),
            "",
            "# Run aggregation once after the array completes:",
            f"# {shlex.quote(args.python_executable)} m4_pair_subset_experiment/aggregate_pair_results.py --experiment-root {shlex.quote(Path(args.experiment_root).as_posix())}",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    experiment_root = Path(args.experiment_root).resolve()
    ensure_experiment_root(experiment_root)
    (experiment_root / "logs").mkdir(parents=True, exist_ok=True)
    pairs_csv = Path(args.pairs_csv).resolve() if args.pairs_csv else pair_manifest_path(experiment_root)
    jobs = selected_jobs(args, pairs_csv)
    pair_ids = [job.pair_id for job in jobs]

    launchers_dir = experiment_root / "launchers"
    launchers_dir.mkdir(parents=True, exist_ok=True)
    ids_path = launchers_dir / f"{args.mode}_pair_ids.txt"
    ids_path.write_text("\n".join(pair_ids) + ("\n" if pair_ids else ""), encoding="utf-8")

    output_path = launchers_dir / f"launch_{args.mode}_{args.launcher}.sh"
    if args.launcher == "bash":
        write_bash_launcher(args, pair_ids, output_path)
    else:
        write_slurm_launcher(args, pair_ids, output_path)
    print(f"Wrote launcher for {len(pair_ids)} pairs: {output_path}")


if __name__ == "__main__":
    main()
