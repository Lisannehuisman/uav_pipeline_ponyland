# Reproducibility

This file gives a more detailed overview of how the thesis experiments were run and which files are needed to reproduce them.

I mainly use this document to keep track of the original Ponyland setup and to make clear which parts of the project are fully preserved and which parts still depend on external data or model files.

## Original setup

Most experiments were run on the Ponyland GPU cluster.

The main project directory was:

/vol/tensusers6/lisannehuisman

The Python environment that I used was located at:

/vol/tensusers6/lisannehuisman/projects/frcnn/thesis_lies

A snapshot of the installed Python packages is stored in:

thesis_ponyland/environment/ponyland_requirements.txt

The preserved environment contains, among others:

Python 3.10.12
torch 2.5.1+cu121
torchvision 0.20.1+cu121
ultralytics 8.4.9
detectron2 0.6

This is the environment that was still present on Ponyland when I collected the repository files. It should therefore be seen as a useful environment snapshot rather than proof that every experiment throughout the whole thesis period used exactly the same package versions.

## Synthetic dataset

The synthetic benchmark contains 10 classes:

0 tent
1 tank
2 tower
3 container
4 whitevan
5 suv
6 male
7 rock
8 barrel
9 tree

There are 205 object instances and 72 viewpoints per object, giving 14,760 images in total.

The 72 viewpoints come from:

8 azimuths × 3 elevations × 3 radii

The azimuth values are:

0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°

The full M4 split contains:

text
train: 10,332 images
validation: 2,214 images
test: 2,214 images


The exact file lists are stored in:


manifests/train_M1.txt
manifests/train_M2a.txt
manifests/train_M2b.txt
manifests/train_M3.txt
manifests/train_M4.txt
manifests/val.txt
manifests/test.txt


I would use these manifests instead of regenerating the train/validation/test split from scratch.

One important detail is that the original split is image/viewpoint-based and not instance-disjoint. The same underlying object instances can therefore occur in train, validation and test, but from different viewpoints. This was part of the original thesis setup.

## Training regimes

The different regimes only change which viewpoints are available during training.

### M1

elevation = mid
radius = mid
azimuth = 0°

Training images: 138.

### M2a

all elevations
radius = mid
azimuth = 0°

Training images: 422.

### M2b

all elevations
radius = mid
azimuth = 0°, 90°, 180°, 270°

Training images: 1,714.

### M3

elevation = mid
radius = mid
all 8 azimuths

Training images: 1,144.

### M4

All 72 viewpoints are available during training.

Training images: 10,332.

The dataset YAML files are stored in:

thesis_ponyland/configs/yamls/

## YOLOv8l

The main YOLOv8l launcher is:

thesis_ponyland/yolov8l/train_all_regimes.sh

The launcher trains M1, M2a, M2b, M3 and M4.

The important settings are:

model = yolov8l.pt
imgsz = 640
epochs = 100
batch = 16
workers = 8
seed = 0
mosaic = 0
mixup = 0
copy_paste = 0

The preserved `args.yaml` files are stored in:

thesis_ponyland/configs/yolov8l_args/

They also show settings such as:

patience = 100
optimizer = auto
deterministic = true
amp = true
lr0 = 0.01
lrf = 0.01
momentum = 0.937
weight_decay = 0.0005

There is a small naming inconsistency for the M1 run. The physical output folder is called:

S0_M1_yolov8l_run1

while the internal metadata contains a different run-name value. I left this as it was instead of changing historical metadata.

## YOLOv8n

The YOLOv8n training code is:

thesis_ponyland/yolov8n/train_yolov8n.py

The corresponding saved `args.yaml` files are in:

thesis_ponyland/configs/yolov8n_args/

The main settings were:

epochs = 120
imgsz = 640
batch = 16
workers = 4
seed = 0
mosaic = 1.0
mixup = 0
copy_paste = 0

Other settings include:

patience = 100
optimizer = auto
deterministic = true
amp = true
lr0 = 0.01
lrf = 0.01
momentum = 0.937
weight_decay = 0.0005

The YOLOv8n and YOLOv8l settings are therefore not completely identical.

## Faster R-CNN

The Faster R-CNN training code is stored in:

thesis_ponyland/frcnn/

The model is based on the Detectron2 configuration:

COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml

The main settings in the preserved code are:

IMS_PER_BATCH = 2
BASE_LR = 0.00025
MAX_ITER = 1500
STEPS = []
WARMUP_ITERS = 0
TEST.EVAL_PERIOD = 500
ROI_HEADS.NUM_CLASSES = 10
ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128

The model was initialized from the corresponding Detectron2 model-zoo weights.

The shell scripts contain scheduler directives, but I did not find enough evidence to state with certainty how every historical job was submitted. For that reason I leave the launcher files as they are instead of describing a scheduler command that I cannot verify.

## Detector comparison

The detector comparison code is in:

thesis_ponyland/detector_comparison/code/

The comparison includes:

YOLOv8n
YOLOv8l
Faster R-CNN

for all five regimes:

M1
M2a
M2b
M3
M4

The final corrected result file is:

thesis_ponyland/detector_comparison/results/standardized_test_summary_CORRECTED.csv

The older version is kept as:

standardized_test_summary_PRE_CORRECTION.csv

### Precision, recall and F1 correction

The first detector evaluation script produced the COCO AP metrics correctly, but the precision, recall and F1 values were calculated incorrectly.

The old implementation averaged COCO precision and recall arrays at IoU 0.5. Later I corrected this by calculating TP, FP and FN with one-to-one matching.

The corrected evaluation uses:

IoU threshold = 0.50
maximum detections per image = 100

For every detector/regime combination, the confidence threshold is selected on the validation set by maximizing F1. That threshold is then kept fixed for the test set.

The correction script is:

thesis_ponyland/detector_comparison/code/recalculate_detector_prf1.py

For final detector comparison numbers, I would therefore use the corrected summary file.

## Main M4 checkpoint

The SHA-256 hash of the original synthetic YOLOv8l-M4 `best.pt` checkpoint is:

a44f9235a84f0d5a21fce65044a322feed993ddb95b795cbc04d8da03227339a

A later M4 reproduction run also exists. Its hash is:

425bf7cb881a36b3e96912e66a79963126951ac89242e807298e75910bf413da

These are not the same checkpoint. The later run also differs in some training settings, including mosaic augmentation, so it should not simply be substituted for the original M4 model.

## Single-viewpoint experiment

The code is stored in:

thesis_ponyland/viewpoint_training/code/single/

The results are in:

thesis_ponyland/viewpoint_training/results/single/

There are 72 single-viewpoint models, one for each viewpoint in the M4 grid.

All 72 runs completed.

Each model was trained on one viewpoint and then evaluated on the full fixed M4 test set.

The preserved result table has a mean mAP50:95 of approximately:

0.33838

The best run is:

sv0034
viewpoint: elmid-radmid-az045
mAP50:95 ≈ 0.41641


## Pair-viewpoint experiment

The code is stored in:

thesis_ponyland/viewpoint_training/code/pair/

The result table is in:

thesis_ponyland/viewpoint_training/results/pair/

With 72 viewpoints there are:

C(72,2) = 2,556 possible pairs


Each pair experiment trains one normal YOLOv8l detector using images from two selected training viewpoints.

Two evaluation options were used:

Option A: evaluate on the full 72-view M4 test set
Option B: evaluate only on the two selected test viewpoints

Option A is the more useful comparison because it tests generalization to the complete viewpoint distribution.

The mean Option-A mAP50:95 is approximately:

0.42066


The best pair is:

p0569
ellow-radmid-az000
elmid-radmid-az225


with an Option-A mAP50:95 of approximately:


0.49577


## Matched controls

The matched-control code is in:

thesis_ponyland/viewpoint_training/code/matched_controls/

and the result table is in:

thesis_ponyland/viewpoint_training/results/matched_controls/

These runs were added to compare the single/pair viewpoint experiments with control subsets of similar size.

The preserved table contains four completed controls:

single best
single mean
pair best
pair mean

Again, the mAP values are the main values I would use from these historical result files.

## Historical multi-view scripts

The earlier multi-view code is in:

thesis_ponyland/multiview/

This includes scripts such as:

score_single_complete.py
score_pairs_true2view.py
score_triples_full_72.py

I kept these because they are part of the development process, but they should be interpreted carefully.

For example, the older pair analysis uses a best-of-view score based on the maximum of two single-view scores. This is not the same thing as actual detector fusion.

Some of the large historical output files also showed class/viewpoint alignment issues. The very large generated CSVs are therefore not included in the repository.

## Operational multi-view analysis

A comparison script is stored at:

thesis_ponyland/viewpoint_training/operational_analysis/run_comparison.py

This script expects an operational protocol summary file from an upstream analysis.

I could not find the original script that generated that summary anymore on Ponyland.

For that reason this part is not fully reproducible from the repository as it currently stands. The surviving comparison script is still included, but any future reconstruction of the missing upstream step should be clearly marked as reconstructed.

## Real transfer

The real-transfer scripts are in:

thesis_ponyland/real_transfer/

They contain conversion/preparation code for real UAV datasets and scripts used during the synthetic-to-real evaluation.

The actual third-party datasets are not included in this repository.

## Real UAV fine-tuning

The real UAV fine-tuning scripts are in:

thesis_ponyland/real_uav_finetune/

The associated dataset configuration is stored in:

thesis_ponyland/configs/real_uav/

The converted real UAV dataset contained 156 images.

The main fine-tuning launcher is:


run_real_uav_finetune_lowLR.sh


The most important settings are:

base model = original synthetic YOLOv8l-M4 best.pt
epochs = 60
patience = 20
imgsz = 640
batch = 8
workers = 8
optimizer = AdamW
lr0 = 0.0005
lrf = 0.01
weight_decay = 0.0005
warmup_epochs = 3
seed = 0
deterministic = true
amp = true
cache = false
mosaic = 0.2
mixup = 0.0
close_mosaic = 10

The SHA-256 hash of the final fine-tuned checkpoint is:

bfac13a3260fa4e5d1ad678ba9c1f291be20aa0b51dd56e773d628c106619387

The model lineage is:

YOLOv8l pretrained weights
→ synthetic M4 training
→ synthetic M4 best.pt
→ real UAV fine-tuning
→ real UAV fine-tuned best.pt

## Files needed for a full rerun

The repository does not contain all files needed for a complete rerun.

For training, the following external files are still needed:

synthetic images
YOLO labels
COCO annotations where needed
real UAV images/labels

For evaluation-only reproduction, the trained model checkpoints are also needed.

These large files were intentionally not pushed to normal Git.

The exact synthetic split membership can be recreated from the manifests in the repository.

## Suggested reproduction order

If I were to rerun the thesis experiments from scratch, I would use roughly this order:

1. obtain the synthetic dataset
2. recreate the exact split using manifests/
3. update the paths in the YAML/config files
4. create the Python environment
5. train YOLOv8l M1-M4
6. train YOLOv8n M1-M4
7. train Faster R-CNN M1-M4
8. run the standardized detector comparison
9. run the corrected P/R/F1 evaluation
10. run the 72 single-viewpoint sweep if needed
11. run the pair-viewpoint sweep if needed
12. run the matched controls
13. prepare the real UAV dataset
14. fine-tune the M4 model on the real UAV data

## Running outside Ponyland

The code under `thesis_ponyland/` is mainly kept in its original form.

Because of that, some scripts still contain hard-coded paths such as:

/vol/tensusers6/lisannehuisman/...

To run the code on another machine, these paths need to be updated or replaced by local paths.

I prefer to keep the original scripts unchanged for now, because they show how the experiments were actually set up. If I later make a cleaner portable version, I would keep that separate from the original `thesis_ponyland/` files.
