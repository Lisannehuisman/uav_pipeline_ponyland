from __future__ import annotations

import argparse
import shlex
from pathlib import Path

from single_experiment_lib import ensure_single_experiment_root, load_single_jobs, single_manifest_path


def shell_value_or_quote(value: str) -> str:
    return value if "${" in value else shlex.quote(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate generic bash or Slurm launch scripts for the M4 single-viewpoint sweep.",
    )
    parser.add_argument("--experiment-root", default="outputs/m4_single_subset_experiment")
    parser.add_argument("--singles-csv", default="", help="Optional explicit single-viewpoint manifest path.")
    parser.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    parser.add_argument("--launcher", choices=["bash", "slurm"], default="bash")
    parser.add_argument("--python-executable", default=".venv/bin/python")
    parser.add_argument("--job-name", default="m4-single-sweep")
    parser.add_argument("--workspace-root", default=str(Path.cwd()))
    parser.add_argument("--base-data-yaml", default="/vol/tensusers6/lisannehuisman/yamls/M4_yolov8l.yaml")
    parser.add_argument("--device", default="0")
    parser.add_argument("--eval-device", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--batch", type=int, default=0)
    parser.add_argument("--imgsz", type=int, default=0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--eval-batch", type=int, default=16)
    parser.add_argument("--resume-training", action="store_true")
    parser.add_argument("--max-parallel", type=int, default=16)
    parser.add_argument("--slurm-partition", default="")
    parser.add_argument("--slurm-account", default="")
    parser.add_argument("--slurm-time", default="")
    parser.add_argument("--slurm-mem", default="")
    parser.add_argument("--slurm-cpus-per-task", type=int, default=0)
    parser.add_argument("--slurm-gres", default="")
    return parser.parse_args()


def selected_jobs(args: argparse.Namespace, singles_csv: Path):
    jobs = load_single_jobs(singles_csv)
    if args.mode == "pilot":
        jobs = [job for job in jobs if job.pilot_rank is not None]
        jobs = sorted(jobs, key=lambda job: job.pilot_rank or 9999)
    return jobs


def build_worker_command(args: argparse.Namespace, single_id: str, device_value: str | None = None) -> str:
    single_value = shell_value_or_quote(single_id)
    resolved_device = device_value if device_value is not None else args.device
    experiment_root = Path(args.experiment_root).as_posix()
    command = [
        shlex.quote(args.python_executable),
        "m4_single_subset_experiment/run_single_experiment.py",
        "--single-id",
        single_value,
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


def write_bash_launcher(args: argparse.Namespace, single_ids: list[str], output_path: Path) -> None:
    lines = ["#!/usr/bin/env bash", *build_shell_preamble(args), "", "SINGLE_IDS=("]
    for single_id in single_ids:
        lines.append(f"  {shlex.quote(single_id)}")
    lines.extend([")", "", 'for SINGLE_ID in "${SINGLE_IDS[@]}"; do'])
    lines.append(f"  {build_worker_command(args, '${SINGLE_ID}')}")
    lines.extend(
        [
            "done",
            "",
            f"{shlex.quote(args.python_executable)} m4_single_subset_experiment/aggregate_single_results.py --experiment-root {shlex.quote(Path(args.experiment_root).as_posix())}",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_slurm_launcher(args: argparse.Namespace, single_ids: list[str], output_path: Path) -> None:
    array_limit = max(0, len(single_ids) - 1)
    log_dir = Path(args.experiment_root).as_posix() + "/logs"
    lines = [
        "#!/usr/bin/env bash",
        f"#SBATCH --job-name={args.job_name}",
        f"#SBATCH --array=0-{array_limit}%{args.max_parallel}",
        f"#SBATCH --output={log_dir}/m4_single_%A_%a.out",
        f"#SBATCH --error={log_dir}/m4_single_%A_%a.err",
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
    lines.extend(["", *build_shell_preamble(args), "", "SINGLE_IDS=("])
    for single_id in single_ids:
        lines.append(f"  {shlex.quote(single_id)}")
    lines.extend(
        [
            ")",
            'SINGLE_ID="${SINGLE_IDS[$SLURM_ARRAY_TASK_ID]}"',
            build_worker_command(args, '${SINGLE_ID}', device_value='${M4_DEVICE}'),
            "",
            "# Run aggregation once after the array completes:",
            f"# {shlex.quote(args.python_executable)} m4_single_subset_experiment/aggregate_single_results.py --experiment-root {shlex.quote(Path(args.experiment_root).as_posix())}",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    experiment_root = Path(args.experiment_root).resolve()
    ensure_single_experiment_root(experiment_root)
    singles_csv = Path(args.singles_csv).resolve() if args.singles_csv else single_manifest_path(experiment_root)
    jobs = selected_jobs(args, singles_csv)
    single_ids = [job.single_id for job in jobs]

    launchers_dir = experiment_root / "launchers"
    launchers_dir.mkdir(parents=True, exist_ok=True)
    ids_path = launchers_dir / f"{args.mode}_single_ids.txt"
    ids_path.write_text("\n".join(single_ids) + ("\n" if single_ids else ""), encoding="utf-8")

    output_path = launchers_dir / f"launch_{args.mode}_{args.launcher}.sh"
    if args.launcher == "bash":
        write_bash_launcher(args, single_ids, output_path)
    else:
        write_slurm_launcher(args, single_ids, output_path)
    print(f"Wrote launcher for {len(single_ids)} single viewpoints: {output_path}")


if __name__ == "__main__":
    main()
