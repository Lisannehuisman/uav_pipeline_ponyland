#!/bin/bash
#SBATCH --job-name=frcnn_%x
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail

TAG="${1:?Usage: sbatch coco_train_template.sh M2a}"

echo "HOSTNAME: $(hostname)"
echo "PWD: $(pwd)"
echo "TAG: $TAG"

cd /vol/tensusers6/lisannehuisman/projects/frcnn
mkdir -p logs

# activate env
source /vol/tensusers6/lisannehuisman/projects/frcnn/thesis_lies/bin/activate

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

TRAIN_JSON="/vol/tensusers6/$USER/data/coco_annotations/coco_instances_train_${TAG}.json"
VAL_JSON="/vol/tensusers6/$USER/data/coco_annotations/coco_instances_val_${TAG}.json"
TRAIN_IMAGES="/vol/tensusers6/$USER/data/images/train_${TAG}"
VAL_IMAGES="/vol/tensusers6/$USER/data/images/val"

OUT="/vol/tensusers6/lisannehuisman/runs/frcnn/S0_${TAG}_run1"
mkdir -p "$OUT"

echo "TRAIN_JSON=$TRAIN_JSON"
echo "VAL_JSON=$VAL_JSON"
echo "TRAIN_IMAGES=$TRAIN_IMAGES"
echo "VAL_IMAGES=$VAL_IMAGES"
echo "OUT=$OUT"

python train_frcnn_m1.py \
  --train_json   "$TRAIN_JSON" \
  --val_json     "$VAL_JSON" \
  --train_images "$TRAIN_IMAGES" \
  --val_images   "$VAL_IMAGES" \
  --out_dir      "$OUT" \
  --num_classes  10 \
  --batch        2 \
  --workers      4 \
  --lr           0.00025 \
  --max_iter     1500 \
  --eval_period  500
