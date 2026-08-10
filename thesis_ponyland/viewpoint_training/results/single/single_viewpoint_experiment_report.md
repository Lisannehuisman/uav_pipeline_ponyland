# M4 Single-Viewpoint Training Report

## What Was Trained

- One normal single-image YOLOv8l detector per training viewpoint.
- Each model is trained only on M4 `train` images whose filenames match the selected viewpoint.
- Validation during training uses the matching viewpoint-filtered `val` split.
- Labels are preserved exactly by reusing the original M4 label files through YOLO list files.

## Scientific Question

- This experiment asks how much detector generalization can be learned from training on only one viewpoint.
- It provides a matched single-view baseline for the duo-viewpoint training sweep.

## Evaluation Protocol

- Every viewpoint-trained model is evaluated on the full fixed M4 test split across all 72 viewpoints.
- This measures generalization from a restricted training subset to the full viewpoint space.

## Current Sweep Status

- Viewpoint definitions: 72
- Completed evaluations: 72

## Best Completed Training Viewpoint

- Viewpoint id: `sv0034`
- Viewpoint: `elmid-radmid-az045` (mid | mid | az045)
- `mAP50-95`: 0.4164
- `mAP50`: 0.6216
- `F1`: 0.6558

## Worst Completed Training Viewpoint

- Viewpoint id: `sv0007`
- Viewpoint: `ellow-radnear-az270` (low | near | az270)
- `mAP50-95`: 0.2678
- `mAP50`: 0.4419
- `F1`: 0.4998

## Pilot Viewpoints

- Pilot 1: `sv0017` = `ellow-radfar-az000` (low_far_front)
- Pilot 2: `sv0011` = `ellow-radmid-az090` (low_mid_side)
- Pilot 3: `sv0037` = `elmid-radmid-az180` (mid_mid_back)
- Pilot 4: `sv0059` = `elhigh-radmid-az090` (high_mid_side)
- Pilot 5: `sv0055` = `elhigh-radnear-az270` (high_near_back)
