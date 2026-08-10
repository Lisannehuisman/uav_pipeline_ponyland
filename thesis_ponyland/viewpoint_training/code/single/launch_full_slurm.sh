#!/usr/bin/env bash
#SBATCH --job-name=m4-single-sweep
#SBATCH --array=0-71%8
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
  sv0001
  sv0002
  sv0003
  sv0004
  sv0005
  sv0006
  sv0007
  sv0008
  sv0009
  sv0010
  sv0011
  sv0012
  sv0013
  sv0014
  sv0015
  sv0016
  sv0017
  sv0018
  sv0019
  sv0020
  sv0021
  sv0022
  sv0023
  sv0024
  sv0025
  sv0026
  sv0027
  sv0028
  sv0029
  sv0030
  sv0031
  sv0032
  sv0033
  sv0034
  sv0035
  sv0036
  sv0037
  sv0038
  sv0039
  sv0040
  sv0041
  sv0042
  sv0043
  sv0044
  sv0045
  sv0046
  sv0047
  sv0048
  sv0049
  sv0050
  sv0051
  sv0052
  sv0053
  sv0054
  sv0055
  sv0056
  sv0057
  sv0058
  sv0059
  sv0060
  sv0061
  sv0062
  sv0063
  sv0064
  sv0065
  sv0066
  sv0067
  sv0068
  sv0069
  sv0070
  sv0071
  sv0072
)
SINGLE_ID="${SINGLE_IDS[$SLURM_ARRAY_TASK_ID]}"
/vol/tensusers6/lisannehuisman/projects/frcnn/thesis_lies/bin/python m4_single_subset_experiment/run_single_experiment.py --single-id ${SINGLE_ID} --base-data-yaml /vol/tensusers6/lisannehuisman/yamls/M4_yolov8l.yaml --experiment-root /vol/tensusers6/lisannehuisman/experiments/m4_single_viewpoint_training --stage all --device ${M4_DEVICE} --eval-batch 16 --model /vol/tensusers6/lisannehuisman/projects/yolov8l.pt --resume-training

# Run aggregation once after the array completes:
# /vol/tensusers6/lisannehuisman/projects/frcnn/thesis_lies/bin/python m4_single_subset_experiment/aggregate_single_results.py --experiment-root /vol/tensusers6/lisannehuisman/experiments/m4_single_viewpoint_training
