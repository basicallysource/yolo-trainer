import argparse
import json
import os
import time
from typing import TypedDict, Optional


class Config(TypedDict):
    yolo_base_model_path: str
    yolo_model: str
    checkpoints_dir: str
    current_run_id: str
    data_yaml_path: str
    data_path: str
    epochs: int
    batch_size: int
    device: str
    val_split: Optional[float]
    img_size: int


def build_config() -> Config:
    parser = argparse.ArgumentParser(description="YOLO segmentation fine-tuning")
    parser.add_argument(
        "config_path",
        type=str,
        help="Path to JSON config file",
    )
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        json_config = json.load(f)

    model_size = json_config.get("model_size", "small")
    model_size_map = {
        "nano": ("yolo11n-seg.pt", "yolo11n-seg"),
        "small": ("yolo11s-seg.pt", "yolo11s-seg"),
        "medium": ("yolo11m-seg.pt", "yolo11m-seg")
    }
    model_file, model_name = model_size_map[model_size]

    checkpoints_dir = json_config.get("checkpoints_dir", "checkpoints")
    checkpoint_run_id = json_config.get("checkpoint_run_id")
    epochs = json_config.get("epochs", 100)
    batch_size = json_config.get("batch_size", 20)
    img_size = json_config.get("img_size", 416)
    data_path = json_config["data_path"]

    if checkpoint_run_id:
        current_run_id = checkpoint_run_id
        print(f"Resuming training from checkpoint run: {current_run_id}")
    else:
        timestamp = int(time.time())
        data_path_clean = data_path.replace("/", "_")
        current_run_id = f"run_{timestamp}_{img_size}_{model_size}_{epochs}epochs_{batch_size}batch_{data_path_clean}"
        print(f"Starting new training run: {current_run_id}")

    os.makedirs(os.path.join(checkpoints_dir, current_run_id), exist_ok=True)

    return {
        "yolo_base_model_path": f"weights/{model_file}",
        "yolo_model": model_name,
        "checkpoints_dir": checkpoints_dir,
        "current_run_id": current_run_id,
        "data_yaml_path": json_config["data_yaml_path"],
        "data_path": data_path,
        "epochs": epochs,
        "batch_size": batch_size,
        "device": json_config.get("device", "cuda"),
        "val_split": json_config.get("val_split"),
        "img_size": img_size,
    }


# Class mapping for visualization
CLASS_NAMES = {
    0: "object",
    1: "first_feeder",
    2: "second_feeder",
    3: "main_conveyor",
    4: "feeder_conveyor"
}
