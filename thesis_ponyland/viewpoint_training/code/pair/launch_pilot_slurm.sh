#!/usr/bin/env bash
#SBATCH --job-name=m4-pair-sweep
#SBATCH --array=0-4%5
#SBATCH --output=/vol/tensusers6/lisannehuisman/experiments/m4_yolov8l_pair_training/logs/m4_pair_%A_%a.out
#SBATCH --error=/vol/tensusers6/lisannehuisman/experiments/m4_yolov8l_pair_training/logs/m4_pair_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --time=06:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

set -euo pipefail
cd /vol/tensusers6/lisannehuisman/projects/m4_yolov8l_pair_training
mkdir -p /vol/tensusers6/lisannehuisman/experiments/m4_yolov8l_pair_training/logs
export PYTHONUNBUFFERED=1
export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-/vol/tensusers6/lisannehuisman/projects/m4_yolov8l_pair_training/Ultralytics}"
mkdir -p "$YOLO_CONFIG_DIR"

PAIR_IDS=(
  p1017
  p0713
  p2419
  p1780
  p1052
)
PAIR_ID="${PAIR_IDS[$SLURM_ARRAY_TASK_ID]}"
/vol/tensusers6/lisannehuisman/projects/frcnn/thesis_lies/bin/python m4_pair_subset_experiment/run_pair_experiment.py --pair-id ${PAIR_ID} --base-data-yaml /vol/tensusers6/lisannehuisman/yamls/M4_yolov8l.yaml --experiment-root /vol/tensusers6/lisannehuisman/experiments/m4_yolov8l_pair_training --stage all --device 0 --eval-batch 8 --model /vol/tensusers6/lisannehuisman/projects/yolov8l.pt --batch 4 --resume-training

# Run aggregation once after the array completes:
# /vol/tensusers6/lisannehuisman/projects/frcnn/thesis_lies/bin/python m4_pair_subset_experiment/aggregate_pair_results.py --experiment-root /vol/tensusers6/lisannehuisman/experiments/m4_yolov8l_pair_training
