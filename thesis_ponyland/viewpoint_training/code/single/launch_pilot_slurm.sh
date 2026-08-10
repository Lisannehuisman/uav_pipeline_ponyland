#!/usr/bin/env bash
#SBATCH --job-name=m4-single-sweep
#SBATCH --array=0-4%5
#SBATCH --output=/vol/tensusers6/lisannehuisman/experiments/m4_single_viewpoint_training/logs/m4_single_%A_%a.out
#SBATCH --error=/vol/tensusers6/lisannehuisman/experiments/m4_single_viewpoint_training/logs/m4_single_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

set -euo pipefail
cd /vol/tensusers6/lisannehuisman/projects/m4_yolov8l_pair_training
mkdir -p /vol/tensusers6/lisannehuisman/experiments/m4_single_viewpoint_training/logs
export PYTHONUNBUFFERED=1
export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-/vol/tensusers6/lisannehuisman/projects/m4_yolov8l_pair_training/Ultralytics}"
mkdir -p "$YOLO_CONFIG_DIR"
M4_DEVICE="${SLURM_STEP_GPUS:-}"
if [ -z "$M4_DEVICE" ] && [ -n "${SLURM_JOB_GPUS:-}" ]; then M4_DEVICE="${SLURM_JOB_GPUS%%,*}"; fi
if [ -n "$M4_DEVICE" ]; then M4_DEVICE="${M4_DEVICE%%,*}"; fi
if [ -z "$M4_DEVICE" ] && [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then M4_DEVICE="0"; fi
if [ -z "$M4_DEVICE" ]; then M4_DEVICE=0; fi
echo "Resolved M4 device: $M4_DEVICE (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}, SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-unset}, SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset})"

SINGLE_IDS=(
  sv0017
  sv0011
  sv0037
  sv0059
  sv0055
)
SINGLE_ID="${SINGLE_IDS[$SLURM_ARRAY_TASK_ID]}"
/vol/tensusers6/lisannehuisman/projects/frcnn/thesis_lies/bin/python m4_single_subset_experiment/run_single_experiment.py --single-id ${SINGLE_ID} --base-data-yaml /vol/tensusers6/lisannehuisman/yamls/M4_yolov8l.yaml --experiment-root /vol/tensusers6/lisannehuisman/experiments/m4_single_viewpoint_training --stage all --device ${M4_DEVICE} --eval-batch 16 --model /vol/tensusers6/lisannehuisman/projects/yolov8l.pt --resume-training

# Run aggregation once after the array completes:
# /vol/tensusers6/lisannehuisman/projects/frcnn/thesis_lies/bin/python m4_single_subset_experiment/aggregate_single_results.py --experiment-root /vol/tensusers6/lisannehuisman/experiments/m4_single_viewpoint_training
