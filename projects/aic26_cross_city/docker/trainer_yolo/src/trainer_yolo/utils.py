import types
from pathlib import Path
from typing import Optional

import mlflow
from hafnia.experiment import HafniaLogger
from hafnia.log import user_logger
from ultralytics import YOLO

CLI_TOOL = "cyclopts"

CLASS_MAPPINGS = {
    "COCO2OnlyVehicle": {
        "bicycle": "Vehicle",
        "car": "Vehicle",
        "motorcycle": "Vehicle",
        "bus": "Vehicle",
        "truck": "Vehicle",
        # Ignore the remaining classes
    },
    "Midwest2OnlyVehicle": {
        "Vehicle*": "Vehicle",
        # Ignore the remaining classes
    },
}

# Updated to map Ultralytics YOLO metrics to Hafnia's expected MLflow structure
_METRIC_KEY_MAP = {
    "train/box_loss": "Loss/Train/Box",
    "train/cls_loss": "Loss/Train/Cls",
    "train/dfl_loss": "Loss/Train/DFL",
    "val/box_loss": "Loss/Test/Box",
    "val/cls_loss": "Loss/Test/Cls",
    "val/dfl_loss": "Loss/Test/DFL",
    "metrics/mAP50-95(B)": "Metrics/Base/AP50_90",
    "metrics/mAP50(B)": "Metrics/Base/AP50",
    # Using YOLO's recall metric as a proxy for AR since native mAR is not directly logged
    "metrics/recall(B)": "Metrics/Base/AR50_90", 
}


def _log_hafnia_metrics(trainer):
    """
    YOLO callback to log metrics to MLflow at the end of each epoch.
    """
    epoch = trainer.epoch
    
    # Ultralytics metric keys often contain padding spaces (e.g., '   val/box_loss')
    # We strip the keys to ensure they match exactly with our _METRIC_KEY_MAP
    cleaned_metrics = {k.strip(): v for k, v in trainer.metrics.items()}
    
    for yolo_key, hafnia_key in _METRIC_KEY_MAP.items():
        value = cleaned_metrics.get(yolo_key)
        if value is not None:
            mlflow.log_metric(hafnia_key, float(value), step=epoch)


def patch_to_support_experiment_tracker_with_hafnia(model: YOLO) -> YOLO:
    """
    Attaches a custom callback to the Ultralytics YOLO model to log Hafnia-specific 
    metrics to MLflow during training.
    """
    # YOLO allows native injection of callbacks at various stages of the training loop
    model.add_callback("on_fit_epoch_end", _log_hafnia_metrics)
    return model


def get_checkpoint_if_available(logger: HafniaLogger) -> Optional[Path]:
    """Return the path to a user-selected checkpoint archive, or ``None`` if none is available.

    On the Hafnia platform a checkpoint selected for an experiment is placed in the checkpoints
    directory (see ``HafniaLogger.path_model_checkpoints``). A checkpoint is a single compressed
    model archive (see ``InitModelConfig.save_model``), so only ``*.zip`` files are considered.
    """
    checkpoints_folder_path = logger.path_model_checkpoints()

    msg_no_checkpoint = "No checkpoint was found. Using pretrained model."
    if not checkpoints_folder_path.exists():
        user_logger.info(msg_no_checkpoint)
        return None

    checkpoint_files = sorted(checkpoints_folder_path.glob("*.zip"))
    if len(checkpoint_files) == 0:
        user_logger.info(msg_no_checkpoint)
        return None

    if len(checkpoint_files) > 1:
        checkpoint_names = [f.name for f in checkpoint_files]
        user_logger.warning(
            f"Only one checkpoint is expected, but multiple were found: {checkpoint_names}. "
            f"Using '{checkpoint_files[0].name}'."
        )

    return checkpoint_files[0]