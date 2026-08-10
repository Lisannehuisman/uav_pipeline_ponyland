import json
import os
from pathlib import Path


REGIME_ORDER = ["M1", "M2a", "M2b", "M3", "M4"]
DETECTOR_ORDER = ["YOLOv8n", "YOLOv8l", "Faster R-CNN"]


MODEL_RUNS = {
    "YOLOv8l": {
        "M1": r"C:\DATA\airsim\thesis\results\yolov8l\yolov8l_results\S0_M1_yolov8l",
        "M2a": r"C:\DATA\airsim\thesis\results\yolov8l\yolov8l_results\S0_M2a_yolov8l",
        "M2b": r"C:\DATA\airsim\thesis\results\yolov8l\yolov8l_results\S0_M2b_yolov8l",
        "M3": r"C:\DATA\airsim\thesis\results\yolov8l\yolov8l_results\S0_M3_yolov8l",
        "M4": r"C:\DATA\airsim\thesis\results\yolov8l\yolov8l_results\S0_M4_yolov8l",
    },
    "YOLOv8n": {
        "M1": r"C:\DATA\airsim\thesis\results\yolov8n\S0_M1_yolov8n",
        "M2a": r"C:\DATA\airsim\thesis\results\yolov8n\S0_M2a_yolov8n",
        "M2b": r"C:\DATA\airsim\thesis\results\yolov8n\S0_M2b_yolov8n",
        "M3": r"C:\DATA\airsim\thesis\results\yolov8n\S0_M3_yolov8n",
        "M4": r"C:\DATA\airsim\thesis\results\yolov8n\S0_M4_yolov8n",
    },
    "Faster R-CNN": {
        "M1": r"C:\DATA\airsim\thesis\results\frcnn\S0_M1_run1",
        "M2a": r"C:\DATA\airsim\thesis\results\frcnn\S0_M2a_run1",
        "M2b": r"C:\DATA\airsim\thesis\results\frcnn\S0_M2b_run1",
        "M3": r"C:\DATA\airsim\thesis\results\frcnn\S0_M3_run1",
        "M4": r"C:\DATA\airsim\thesis\results\frcnn\S0_M4_run1",
    },
}


# Fill these in with the actual dataset YAMLs for the five regimes before running
# standardized_test_eval.py. The script expects the YAMLs to define the test split.
REGIME_DATA_YAMLS = {
    "M1": r"C:\DATA\airsim\thesis\captures\S0_20251219_164144\dataset\M1.yaml",
    "M2a": r"C:\DATA\airsim\thesis\captures\S0_20251219_164144\dataset\M2a.yaml",
    "M2b": r"C:\DATA\airsim\thesis\captures\S0_20251219_164144\dataset\M2b.yaml",
    "M3": r"C:\DATA\airsim\thesis\captures\S0_20251219_164144\dataset\M3.yaml",
    "M4": r"C:\DATA\airsim\thesis\captures\S0_20251219_164144\dataset\M4_fixed.yaml",
}


DEFAULT_OUTPUT_DIR = Path("outputs") / "detector_family_comparison"


def _load_override_config() -> None:
    override_path = os.environ.get("DETECTOR_COMPARISON_CONFIG_JSON")
    if not override_path:
        return

    config_path = Path(override_path).expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))

    global REGIME_ORDER, DETECTOR_ORDER, MODEL_RUNS, REGIME_DATA_YAMLS, DEFAULT_OUTPUT_DIR

    if "regime_order" in config:
        REGIME_ORDER = list(config["regime_order"])
    if "detector_order" in config:
        DETECTOR_ORDER = list(config["detector_order"])
    if "model_runs" in config:
        MODEL_RUNS = {
            str(detector): {str(regime): str(path) for regime, path in runs.items()}
            for detector, runs in config["model_runs"].items()
        }
    if "regime_data_yamls" in config:
        REGIME_DATA_YAMLS = {str(regime): str(path) for regime, path in config["regime_data_yamls"].items()}
    if "default_output_dir" in config:
        DEFAULT_OUTPUT_DIR = Path(config["default_output_dir"])


_load_override_config()
