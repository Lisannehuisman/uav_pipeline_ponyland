import os
import argparse
from detectron2.utils.logger import setup_logger
from detectron2.data.datasets import register_coco_instances
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator

class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, False, output_folder)



def main(args):
    setup_logger()
    print(">>> START: registering datasets", flush=True)

    register_coco_instances("S0_M1_train", {}, args.train_json, args.train_images)
    register_coco_instances("S0_M1_val",   {}, args.val_json,   args.val_images)

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(args.model_cfg))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(args.model_cfg)

    cfg.DATASETS.TRAIN = ("S0_M1_train",)
    cfg.DATASETS.TEST  = ("S0_M1_val",)

    cfg.DATALOADER.NUM_WORKERS = args.workers
    cfg.SOLVER.IMS_PER_BATCH = args.batch
    cfg.SOLVER.BASE_LR = args.lr
    cfg.SOLVER.MAX_ITER = args.max_iter
    cfg.SOLVER.STEPS = []
    cfg.SOLVER.WARMUP_ITERS = 0

    cfg.TEST.EVAL_PERIOD = args.eval_period

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = args.num_classes
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128

    cfg.OUTPUT_DIR = args.out_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    print(">>> CONFIG OK. OUTPUT_DIR =", cfg.OUTPUT_DIR, flush=True)
    print(">>> START TRAINING", flush=True)

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_json", required=True)
    ap.add_argument("--val_json", required=True)
    ap.add_argument("--train_images", required=True)
    ap.add_argument("--val_images", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--num_classes", type=int, required=True)

    ap.add_argument("--model_cfg", default="COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.00025)
    ap.add_argument("--max_iter", type=int, default=1500)
    ap.add_argument("--eval_period", type=int, default=500)

    args = ap.parse_args()
    main(args)
