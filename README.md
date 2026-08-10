# UAV Thesis Repository

This repository contains the code, configuration files, manifests and main result files that I used for my master's thesis on UAV object detection, viewpoint diversity and multi-view analysis.

Most of the original experiments were run on the Ponyland GPU cluster at Radboud University. I wanted to keep the original code available, but at the same time avoid uploading very large datasets, checkpoints and prediction caches to GitHub. Because of that, this repository mainly contains the scripts, configs and compact result files that are needed to understand how the experiments were run.

For more details about reproducing the experiments and data, email lisanne.huisman@ru.nl or lisanne.huisman@outlook.com

## Repository structure

uav_thesis_lisanne/
├── README.md
├── REPRODUCIBILITY.md
├── manifests/
├── results/
└── thesis_ponyland/

The most important folders are:

- `manifests/`: exact train, validation and test file lists that were used for the synthetic dataset.
- `results/`: compact result files from the main training runs.
- `thesis_ponyland/`: original scripts and configs copied from the Ponyland cluster.

Inside `thesis_ponyland/` the code is split into the main experiment parts:

thesis_ponyland/
├── configs/
├── detector_comparison/
├── environment/
├── frcnn/
├── multiview/
├── real_transfer/
├── real_uav_finetune/
├── viewpoint_training/
├── yolov8l/
└── yolov8n/


## Main experiments

The synthetic dataset contains 10 classes:

tent
tank
tower
container
whitevan
suv
male
rock
barrel
tree

Each object was rendered from 72 viewpoints, based on 8 azimuths, 3 elevations and 3 radii.

The main training regimes were:

| Regime | Training viewpoint selection |
|---|---|
| M1 | mid elevation, mid radius, azimuth 0° |
| M2a | all elevations, mid radius, azimuth 0° |
| M2b | all elevations, mid radius, azimuths 0°, 90°, 180°, 270° |
| M3 | mid elevation, mid radius, all 8 azimuths |
| M4 | all 72 viewpoints |

The original YOLOv8l training launcher is here:

thesis_ponyland/yolov8l/train_all_regimes.sh

The YOLOv8n training code is here:

thesis_ponyland/yolov8n/train_yolov8n.py

The Faster R-CNN training code is in:

thesis_ponyland/frcnn/

## Detector comparison

The code used to compare YOLOv8n, YOLOv8l and Faster R-CNN is in:

thesis_ponyland/detector_comparison/

The final corrected detector summary is:

thesis_ponyland/detector_comparison/results/standardized_test_summary_CORRECTED.csv

This corrected file should be used instead of the older pre-correction summary. The original precision, recall and F1 calculation was later found to be incorrect because it averaged COCO evaluation arrays instead of computing TP/FP/FN at a fixed confidence threshold.

The correction script is:

thesis_ponyland/detector_comparison/code/recalculate_detector_prf1.py

## Single-view and pair-view training

The code for the additional viewpoint-training experiments is in:

thesis_ponyland/viewpoint_training/

This includes:

- 72 single-viewpoint training runs;
- all 2,556 possible pairs of the 72 viewpoints;
- matched-data control experiments.

For the pair sweep, 2,535 out of 2,556 runs completed successfully.

The `master_results.csv` files contain the main results from these experiments. Their mAP50 and mAP50:95 values are useful, but the old precision/recall/F1 columns were calculated with the same earlier method mentioned above, so I do not treat those columns as the final corrected P/R/F1 values.

## Real UAV experiments

The real transfer scripts are in:

thesis_ponyland/real_transfer/

The fine-tuning code for the self-collected real UAV dataset is in:

thesis_ponyland/real_uav_finetune/

The real UAV fine-tuning started from the best synthetic YOLOv8l-M4 checkpoint and then fine-tuned that model on the real UAV dataset.

## Multi-view code

The folder

thesis_ponyland/multiview/

contains earlier multi-view experiments and diagnostic scripts.

I kept these files because they are part of the work that led to the final thesis, but not every script in this folder represents the final method. Some of the earlier pair/triple analyses use best-of-view or oracle-style scoring instead of actual detector fusion.

There is also an operational comparison script in:

thesis_ponyland/viewpoint_training/operational_analysis/

The upstream script that originally generated one of the operational summary files could not be found anymore on Ponyland. I therefore keep the surviving comparison script, but I do not present that missing part as fully reproducible.

## Files that are not included

I did not add the following large files to normal Git:

- image datasets;
- YOLO `.pt` checkpoints;
- Faster R-CNN `.pth` checkpoints;
- virtual environments;
- prediction caches;
- thousands of intermediate JSON files;
- very large pair/triple CSV files.

These files are too large for a normal source-code repository and some datasets also have their own redistribution conditions. Large model checkpoints are not tracked as normal Git files. The main trained checkpoints are distributed separately with the repository so that the evaluation experiments can be reproduced without retraining. Python virtual environments are not distributed; the preserved environment requirements and version information are provided instead.

## Original Ponyland paths

Some scripts still contain paths such as:

/vol/tensusers6/lisannehuisman/...

I left these in the original Ponyland scripts on purpose. They show how the experiments were actually run on the cluster.

This means that cloning the repository on another computer is not enough to run every script immediately. Dataset and checkpoint paths first need to be changed to the local setup.

## Environment

A `pip freeze` snapshot from the Ponyland environment is stored in:

thesis_ponyland/environment/ponyland_requirements.txt

Some of the main versions in that environment were:

Python 3.10.12
PyTorch 2.5.1+cu121
torchvision 0.20.1+cu121
Ultralytics 8.4.9
Detectron2 0.6

