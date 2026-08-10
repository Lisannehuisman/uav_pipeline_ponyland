#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash detector_family_comparison/run_test_pipeline_gpu.sh <config-json> [yolo-device] [frcnn-device] [yolo-batch]"
  exit 1
fi

CONFIG_JSON="$1"
YOLO_DEVICE="${2:-0}"
FRCNN_DEVICE="${3:-cuda}"
YOLO_BATCH="${4:-8}"

export DETECTOR_COMPARISON_CONFIG_JSON="$CONFIG_JSON"
export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-/tmp/${USER:-user}_ultralytics}"
mkdir -p "$YOLO_CONFIG_DIR"

python detector_family_comparison/standardized_test_eval.py \
  --split test \
  --detectors YOLOv8n YOLOv8l \
  --regimes M1 M2a M2b M3 M4 \
  --device "$YOLO_DEVICE" \
  --batch "$YOLO_BATCH" \
  --overwrite

python detector_family_comparison/generate_frcnn_predictions.py \
  --split test \
  --regimes M1 M2a M2b M3 M4 \
  --device "$FRCNN_DEVICE" \
  --resume

python detector_family_comparison/standardized_test_eval.py \
  --split test \
  --detectors "Faster R-CNN" \
  --regimes M1 M2a M2b M3 M4 \
  --overwrite

python detector_family_comparison/generate_test_reports.py \
  --split test \
  --per-class-metrics ap50_95 ap50 ap75
