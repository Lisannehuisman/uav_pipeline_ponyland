#!/bin/bash
#SBATCH --job-name=frcnn_S0_M1
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail

# Go to project folder

source /vol/tensusers6/lisannehuisman/projects/frcnn/thesis_lies/bin/activate
# Quick debug info (will appear in the logs
echo "HOSTNAME: $(hostname)"
echo "PWD: $(pwd)"

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

# Dataset root (the folder you upload)
DATA=/vol/tensusers6/lisannehuisman/data

# Output folder (results + model checkpoints)
OUT=/vol/tensusers6/lisannehuisman/runs/frcnn/S0_M1_run1
mkdir -p "$OUT"

python train_frcnn_m1.py \
  --train_json   "$DATA/coco_annotations/coco_instances_train_M1.json" \
  --val_json     "$DATA/coco_annotations/coco_instances_val_M1.json" \
  --train_images "$DATA/images/train_M1" \
  --val_images   "$DATA/images/val" \
  --out_dir      "$OUT" \
  --num_classes  10 \
  --batch        2 \
  --workers      4 \
  --lr           0.00025 \
  --max_iter     1500 \
  --eval_period  500
