import types
from pathlib import Path
from typing import Optional

import mlflow
from hafnia.experiment import HafniaLogger
from hafnia.log import user_logger

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

_METRIC_KEY_MAP = {
    "train/loss": "Loss/Train",
    "val/loss": "Loss/Test",
    "val/mAP_50_95": "Metrics/Base/AP50_90",
    "val/mAP_50": "Metrics/Base/AP50",
    "val/mAR": "Metrics/Base/AR50_90",
    "val/ema_mAP_50_95": "Metrics/EMA/AP50_90",
    "val/ema_mAP_50": "Metrics/EMA/AP50",
    "val/ema_mAR": "Metrics/EMA/AR50_90",
}


def patch_to_support_experiment_tracker_with_hafnia(detr: types.ModuleType):
    import rfdetr.training as _rfdetr_training
    from pytorch_lightning import Callback

    class _HafniaMLflowCallback(Callback):
        def on_validation_epoch_end(self, trainer, pl_module):
            epoch = trainer.current_epoch
            for ptl_key, hafnia_key in _METRIC_KEY_MAP.items():
                value = trainer.callback_metrics.get(ptl_key)
                if value is not None:
                    mlflow.log_metric(hafnia_key, float(value), step=epoch)

    _original_build_trainer = _rfdetr_training.build_trainer

    def _patched_build_trainer(train_config, model_config, **kwargs):
        ptl_trainer = _original_build_trainer(train_config, model_config, **kwargs)
        ptl_trainer.callbacks.append(_HafniaMLflowCallback())
        return ptl_trainer

    _rfdetr_training.build_trainer = _patched_build_trainer
    return detr


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
