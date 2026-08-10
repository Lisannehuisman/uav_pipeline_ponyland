#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-/vol/tensusers6/lisannehuisman}"
OUT_JSON="${2:-detector_family_comparison/ponyland_config.json}"

PROJECTS_DIR="$ROOT_DIR/projects"
YAMLS_DIR="$ROOT_DIR/yamls"
OUTPUT_DIR="$PROJECTS_DIR/compare_yolov8l_frcnn/outputs/detector_family_comparison"

find_first_matching_parent() {
  local search_root="$1"
  local filename="$2"
  local required_a="$3"
  local required_b="$4"
  local strip_levels="$5"
  local match=""
  while IFS= read -r candidate; do
    local lowered
    lowered=$(printf '%s' "$candidate" | tr '[:upper:]' '[:lower:]')
    if [[ "$lowered" == *"$required_a"* && "$lowered" == *"$required_b"* ]]; then
      match="$candidate"
      break
    fi
  done < <(find "$search_root" -name "$filename" -print | sort)

  if [[ -z "$match" ]]; then
    return 1
  fi

  local parent="$match"
  local level
  for ((level=0; level<strip_levels; level++)); do
    parent=$(dirname "$parent")
  done
  printf '%s\n' "$parent"
}

find_yolo_run() {
  local search_root="$1"
  local regime_token="$2"
  local family_token="$3"
  local weights_path
  weights_path=$(find_first_matching_parent "$search_root" "best.pt" "$regime_token" "$family_token" 2 || true)
  if [[ -z "$weights_path" ]]; then
    echo "Could not find a $family_token run for $regime_token under $search_root" >&2
    return 1
  fi
  printf '%s\n' "$weights_path"
}

find_frcnn_run() {
  local search_root="$1"
  local regime_token="$2"
  local checkpoint_path
  checkpoint_path=$(find_first_matching_parent "$search_root" "model_final.pth" "$regime_token" "frcnn" 1 || true)
  if [[ -z "$checkpoint_path" ]]; then
    echo "Could not find an frcnn run for $regime_token under $search_root" >&2
    return 1
  fi
  printf '%s\n' "$checkpoint_path"
}

YOLOV8L_BASE="$PROJECTS_DIR/yolov8l"
YOLOV8N_BASE="$PROJECTS_DIR/yolov8n"
FRCNN_BASE="$ROOT_DIR"

YOLOV8L_M1=$(find_yolo_run "$YOLOV8L_BASE" "m1" "yolov8l")
YOLOV8L_M2A=$(find_yolo_run "$YOLOV8L_BASE" "m2a" "yolov8l")
YOLOV8L_M2B=$(find_yolo_run "$YOLOV8L_BASE" "m2b" "yolov8l")
YOLOV8L_M3=$(find_yolo_run "$YOLOV8L_BASE" "m3" "yolov8l")
YOLOV8L_M4=$(find_yolo_run "$YOLOV8L_BASE" "m4" "yolov8l")

YOLOV8N_M1=$(find_yolo_run "$YOLOV8N_BASE" "m1" "yolov8n")
YOLOV8N_M2A=$(find_yolo_run "$YOLOV8N_BASE" "m2a" "yolov8n")
YOLOV8N_M2B=$(find_yolo_run "$YOLOV8N_BASE" "m2b" "yolov8n")
YOLOV8N_M3=$(find_yolo_run "$YOLOV8N_BASE" "m3" "yolov8n")
YOLOV8N_M4=$(find_yolo_run "$YOLOV8N_BASE" "m4" "yolov8n")

FRCNN_M1=$(find_frcnn_run "$FRCNN_BASE" "m1")
FRCNN_M2A=$(find_frcnn_run "$FRCNN_BASE" "m2a")
FRCNN_M2B=$(find_frcnn_run "$FRCNN_BASE" "m2b")
FRCNN_M3=$(find_frcnn_run "$FRCNN_BASE" "m3")
FRCNN_M4=$(find_frcnn_run "$FRCNN_BASE" "m4")

mkdir -p "$(dirname "$OUT_JSON")"

cat > "$OUT_JSON" <<EOF
{
  "regime_order": ["M1", "M2a", "M2b", "M3", "M4"],
  "detector_order": ["YOLOv8n", "YOLOv8l", "Faster R-CNN"],
  "default_output_dir": "$OUTPUT_DIR",
  "regime_data_yamls": {
    "M1": "$YAMLS_DIR/M1_yolov8l.yaml",
    "M2a": "$YAMLS_DIR/M2a_yolov8l.yaml",
    "M2b": "$YAMLS_DIR/M2b_yolov8l.yaml",
    "M3": "$YAMLS_DIR/M3_yolov8l.yaml",
    "M4": "$YAMLS_DIR/M4_yolov8l.yaml"
  },
  "model_runs": {
    "YOLOv8l": {
      "M1": "$YOLOV8L_M1",
      "M2a": "$YOLOV8L_M2A",
      "M2b": "$YOLOV8L_M2B",
      "M3": "$YOLOV8L_M3",
      "M4": "$YOLOV8L_M4"
    },
    "YOLOv8n": {
      "M1": "$YOLOV8N_M1",
      "M2a": "$YOLOV8N_M2A",
      "M2b": "$YOLOV8N_M2B",
      "M3": "$YOLOV8N_M3",
      "M4": "$YOLOV8N_M4"
    },
    "Faster R-CNN": {
      "M1": "$FRCNN_M1",
      "M2a": "$FRCNN_M2A",
      "M2b": "$FRCNN_M2B",
      "M3": "$FRCNN_M3",
      "M4": "$FRCNN_M4"
    }
  }
}
EOF

echo "Wrote Ponyland config to: $OUT_JSON"
