#!/usr/bin/env bash
set -euo pipefail

source /vol/tensusers6/lisannehuisman/projects/frcnn/thesis_lies/bin/activate

yolo detect train   model="/vol/tensusers6/lisannehuisman/projects/yolov8l/runs_yolov8l/S0_M4_yolov8l_run1/weights/best.pt"   data="/vol/tensusers6/lisannehuisman/projects/real_uav_finetune/data/real_uav_10class/data.yaml"   epochs=60   patience=20   imgsz=640   batch=8   device=0   workers=8   optimizer=AdamW   lr0=0.0005   lrf=0.01   weight_decay=0.0005   warmup_epochs=3   seed=0   deterministic=True   amp=True   cache=False   mosaic=0.2   mixup=0.0   close_mosaic=10   plots=True   save=True   save_period=10   project="/vol/tensusers6/lisannehuisman/projects/real_uav_finetune/runs"   name="real_uav_finetune_yolov8l_m4_lowLR"   exist_ok=False
