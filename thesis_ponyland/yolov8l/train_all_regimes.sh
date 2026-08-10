#!/bin/bash
set -euo pipefail

# ----------------------------
# ACTIVATE ENVIRONMENT
# ----------------------------
source /vol/tensusers6/lisannehuisman/projects/frcnn/thesis_lies/bin/activate

# ----------------------------
# GPU + ULTRALYTICS SETTINGS
# ----------------------------
export CUDA_VISIBLE_DEVICES=1
export ULTRALYTICS_CONFIG_DIR=/vol/tensusers6/lisannehuisman/projects/yolov8l/.ultralytics
mkdir -p "$ULTRALYTICS_CONFIG_DIR"

echo "HOSTNAME: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi

# ----------------------------
# PATHS
# ----------------------------
PROJECT=/vol/tensusers6/lisannehuisman/projects/yolov8l/runs_yolov8l
YAMLS=/vol/tensusers6/lisannehuisman/yamls

mkdir -p "$PROJECT"

# ----------------------------
# TRAIN ALL REGIMES
# ----------------------------
for TAG in M1 M2a M2b M3 M4; do
  echo "=============================="
  echo "TRAINING YOLOv8l – REGIME $TAG"
  echo "=============================="

  yolo detect train \
    model=yolov8l.pt \
    data=${YAMLS}/${TAG}_yolov8l.yaml \
    imgsz=640 \
    epochs=100 \
    batch=16 \
    workers=8 \
    seed=0 \
    device=0 \
    mosaic=0 \
    mixup=0 \
    copy_paste=0 \
    project=$PROJECT \
    name=S0_${TAG}_yolov8l_run1
done

echo "ALL REGIMES FINISHED ✅"
