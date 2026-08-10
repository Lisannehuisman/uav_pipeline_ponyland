# M4 Pair-Subset Experiment Report

## What Was Trained

- One normal single-image YOLOv8l detector per viewpoint pair.
- Each pair model is trained only on M4 `train` images whose filenames match the selected two viewpoints.
- Validation during training uses the matching pair-filtered `val` split.
- Labels are preserved exactly by reusing the original M4 label files through YOLO list files instead of copying annotations.

## Scientific Question

- This experiment asks how much detector generalization can be learned from training on only two viewpoints.
- It does not measure multi-view inference or fusion; the model architecture remains unchanged and each image is still evaluated independently.

## Evaluation Protocol

- `Option A` (recommended): evaluate every pair-trained model on the full fixed M4 test split across all 72 viewpoints.
- `Option B`: evaluate each pair-trained model only on the same two viewpoints in the fixed M4 test split.
- Recommendation: use `Option A` as the headline comparison because it measures generalization from a restricted training subset to the full viewpoint space.
- `Option B` is still useful as a diagnostic for in-subset fit but should not be treated as the primary scientific result.

## Current Sweep Status

- Pair definitions: 2556
- Completed Option A evaluations: 2535
- Recommended default metric source: option_a_full_test

## Best Completed Pair So Far

- Pair: `p0569`
- Viewpoints: `ellow-radmid-az000` + `elmid-radmid-az225`
- Option A `mAP50-95`: 0.4958
- Option A `mAP50`: 0.7252
- Option A `F1`: 0.7478

## Pilot Pairs

- Pilot 1: `p1017` = `ellow-radfar-az000` + `ellow-radfar-az045` (redundant_neighbor)
- Pilot 2: `p0713` = `ellow-radmid-az090` + `elhigh-radmid-az090` (elevation_only)
- Pilot 3: `p2419` = `elhigh-radnear-az270` + `elhigh-radfar-az270` (radius_only)
- Pilot 4: `p1780` = `elmid-radmid-az000` + `elmid-radmid-az180` (azimuth_only)
- Pilot 5: `p1052` = `ellow-radfar-az000` + `elhigh-radnear-az180` (max_contrast)
