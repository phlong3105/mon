#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Run local:
hafnia trainer create-zip . ; ^
hafnia runc build-local trainer.zip ; ^
hafnia runc launch-local --dataset eccv-cross-city  "python scripts/train.py --epochs 1 --batch_size 1 --inference_sahi"
"""

# Fix mlflow connection errors
import os
os.environ["MLFLOW_SQLALCHEMYSTORE_POOL_SIZE"]    = "300"
os.environ["MLFLOW_SQLALCHEMYSTORE_MAX_OVERFLOW"] = "500"
os.environ["MLFLOW_SQLALCHEMYSTORE_POOL_TIMEOUT"] = "600"
os.environ["MLFLOW_SQLALCHEMYSTORE_POOL_RECYCLE"] = "18000"
os.environ["MLFLOW_SQLALCHEMYSTORE_ECHO"]         = "False"
# noinspection PyUnusedImports
import mlflow

import time
from pathlib import Path
from typing import Annotated, Optional, Type

import polars as pl
import torch
from cyclopts import App, Parameter
from hafnia import utils as hafnia_utils
from hafnia.dataset.benchmark.benchmark import metric_calculations
from hafnia.dataset.benchmark.inference_model import InferenceModel
from hafnia.dataset.dataset_names import (
    SampleField,
    SplitName,
    TASK_NAME_PREDICTIONS_POSTFIX,
)
from hafnia.dataset.hafnia_dataset import HafniaDataset
from hafnia.dataset.hafnia_dataset_types import Sample, TaskInfo
from hafnia.dataset.primitives import Primitive
from hafnia.experiment import HafniaLogger
from hafnia.experiment.command_builder import auto_save_command_builder_schema
from hafnia.log import user_logger
from hafnia.utils import progress_bar
from rfdetr import detr

import trainer_object_detection.wrapped_model
from trainer_object_detection import utils
from trainer_object_detection.wrapped_model import (
    InferenceConfig,
    InitModelConfig,
    WrappedModel,
)

# ==============================================================================
# region CONSTANTS
# ==============================================================================

detr = utils.patch_to_support_experiment_tracker_with_hafnia(detr)

app = App(name="train", help="PyTorch Training")
MODEL_NAME_OPTIONS = [
    f"pretrained_models/{d.name}.zip"
    for d in trainer_object_detection.wrapped_model.MODEL_OPTIONS
]
DEFAULT_INFERENCE_MODEL = "checkpoint_best_ema"
INFERENCE_MODEL_OPTIONS = [
    DEFAULT_INFERENCE_MODEL,
    "checkpoint_best_regular",
    "checkpoint_best_total"
]

# Checkpoints
CKPT_DIR = Path("/opt/recipe/checkpoints")
CHECKPOINTS: dict[str, Path] = {
    DEFAULT_INFERENCE_MODEL  : CKPT_DIR / f"{DEFAULT_INFERENCE_MODEL}.zip",
    "checkpoint_best_regular": CKPT_DIR / "checkpoint_best_regular.zip",
    "checkpoint_best_total"  : CKPT_DIR / "checkpoint_best_total.zip",
}

# Augmentations
AUG_CONFIG = {  #: Aggressive augmentations — for larger datasets (2000+ images).
    "HorizontalFlip": {"p": 0.5},
    "VerticalFlip": {"p": 0.5},
    "Rotate": {"limit": 45, "p": 0.5},
    "Affine": {
        "scale": (0.8, 1.2),
        "translate_percent": (-0.1, 0.1),
        "rotate": (-15, 15),
        "shear": (-5, 5),
        "p": 0.5,
    },
    "ColorJitter": {
        "brightness": 0.2,
        "contrast": 0.2,
        "saturation": 0.2,
        "hue": 0.1,
        "p": 0.5,
    },
    "RandomBrightnessContrast": {
        "brightness_limit": 0.15,
        "contrast_limit": 0.15,
        "p": 0.4,
    },
}

# endregion


# ==============================================================================
# region PREDICTION
# ==============================================================================

def remove_images_with_no_bboxes(
    dataset: HafniaDataset,
    model_primitive: Type[Primitive]
) -> HafniaDataset:
    if not dataset.has_primitive(model_primitive):
        raise ValueError("Dataset does not contain bounding box information.")

    filter_column_name = model_primitive.column_name()
    samples_with_bboxes = dataset.samples.filter(pl.col(filter_column_name).list.len() > 0)
    dataset = dataset.update_samples(samples_with_bboxes)
    return dataset


def get_dataset_task_from_model_primitive(
    dataset: HafniaDataset,
    model_primitive: Type[Primitive],
    task_name: Optional[str] = None,
) -> TaskInfo:
    """Select the dataset task that matches the model primitive type."""

    # Get dataset tasks matching the model primitive
    matching_tasks = dataset.info.get_tasks_by_primitive(model_primitive)
    if len(matching_tasks) == 1:
        matching_task = matching_tasks[0]
        return matching_task

    if len(matching_tasks) == 0:
        available_primitives = [str(t.primitive.__name__) for t in dataset.info.tasks]
        raise ValueError(
            f"The selected model requires the dataset to have '{model_primitive}' annotations. "
            f"However, the dataset only contains the following primitives: {available_primitives}"
        )

    if task_name is None:
        matching_task_names = [t.name for t in matching_tasks]
        raise ValueError(
            f"The dataset contains multiple tasks with the required primitive '{model_primitive}'. "
            f"Please specify which task to use with the '--task_name' flag. "
            f"Matching tasks: {matching_task_names}"
        )

    model_task_info = dataset.info.get_task_by_name(task_name)

    if model_task_info.primitive != model_primitive:
        raise ValueError(f"The specified task '{task_name}' does not have the required primitive '{model_primitive}'.")

    return model_task_info


def run_inference_on_dataset(
    dataset: HafniaDataset,
    model: InferenceModel,
    task_name_prediction_postfix: str = TASK_NAME_PREDICTIONS_POSTFIX,
) -> HafniaDataset:
    model_tasks = [m.model_copy() for m in model.get_model_info().tasks]

    new_task_names = [f"{task.name}{task_name_prediction_postfix}" for task in model_tasks]
    user_logger.info(
        f"Running inference on dataset '{dataset.info.dataset_name}'\n"
        f"- Number of samples: {len(dataset)}\n"
        f"- Model tasks: {[task.name for task in model_tasks]}\n"
        f"- Predictions will be appended to the dataset with new task names:\n"
        f"- Prediction task names: {new_task_names}"
    )

    for model_task in model_tasks:
        model_task.name = f"{model_task.name}{task_name_prediction_postfix}"

    prediction_samples = []
    for dict_sample in progress_bar(dataset, description="Running inference on dataset"):
        sample = Sample(**dict_sample)
        image = sample.read_image()

        predictions = model.predict(image, sample_dict=dict_sample)
        for prediction in predictions:
            prediction.task_name = f"{prediction.task_name}{task_name_prediction_postfix}"
        sample.append_primitives(predictions)
        prediction_samples.append(sample)

    prediction_dataset_info = dataset.info.model_copy(deep=True)
    prediction_dataset_info.tasks.extend(model_tasks)

    dataset_predictions = HafniaDataset.from_samples_list(prediction_samples, info=prediction_dataset_info)
    return dataset_predictions

# endregion


# ==============================================================================
# region MAIN
# ==============================================================================

@app.default
def main(
    project_name: Annotated[str, Parameter(help="Project name for the experiment")] = "Trainer RF-DETR",
    model_path: Annotated[
        str,
        Parameter(
            help=(
                "Path to a compressed (zip) pretrained model used as the training starting point. "
                f"Options: {MODEL_NAME_OPTIONS}"
            )
        ),
    ] = "pretrained_models/RFDETR2XLarge.zip",  # "/pretrained_models/RFDETRNano.zip",
    pretrained: Annotated[bool, Parameter(help="Initialize the model from pretrained weights")] = True,
    epochs: Annotated[int, Parameter(help="Number of epochs to train")] = 10,
    batch_size: Annotated[int, Parameter(help="Batch size for training")] = 2,
    grad_accum_steps: Annotated[
        int,
        Parameter(help="Number of gradient accumulation steps (effective batch size = batch_size * grad_accumulation_steps)"),
    ] = 8,
    lr: Annotated[float, Parameter(help="Learning rate for the optimizer")] = 0.001,
    resolution: Annotated[
        Optional[int],
        Parameter(help="Input resolution (square side in pixels). Defaults to each model's built-in value."),
    ] = None,
    task_name: Annotated[
        Optional[str],
        Parameter(help="Dataset task name used for training. Only required when the dataset has multiple tasks matching the model primitive."),
    ] = None,
    samples: Annotated[
        Optional[int],
        Parameter(help="Number of samples to use for training (omit to use all samples). Use for testing purposes."),
    ] = None,
    stop_early: Annotated[
        bool,
        Parameter(help="Exit before training starts. Can be used to avoid long training times when smoke-testing the pipeline."),
    ] = False,
    run_train: Annotated[bool, Parameter(help="Run training")] = True,
    run_test: Annotated[bool, Parameter(help="Run testing")] = True,
    inference_model_name: Annotated[
        str,
        Parameter(help=f"Checkpoint used for the post-training benchmark on the test split. Options: {INFERENCE_MODEL_OPTIONS}"),
    ] = DEFAULT_INFERENCE_MODEL,
    inference_config: Annotated[
        Optional[InferenceConfig], Parameter(help="Inference configuration used for the post-training benchmark")
    ] = InferenceConfig(compile=True, batch_size=1, threshold=0.05),
    inference_sahi: Annotated[bool, Parameter(help="Use SAHI for inference")] = False,
):
    """Train an RF-DETR object detection model on a Hafnia dataset.

    Loads the dataset (the hidden dataset when running on the Hafnia platform, otherwise the
    small public sample dataset is used when executing locally), initializes an RF-DETR model from
    the compressed model archive pointed to by ``model_path`` (optionally with pretrained weights), converts the
    train/val splits to COCO format and runs RF-DETR training.

    After training, every ``checkpoint_*.pth`` produced by RF-DETR is repackaged as a
    standalone compressed Hafnia model archive (weights + serialized model config bundled into a
    single ``.zip``) under the experiment model and checkpoints folders. The checkpoint selected
    by ``inference_model_name`` is then
    loaded as a ``WrappedModel``, optimized for inference (e.g. ``torch.compile`` when enabled
    via ``inference_config``) and run on the held-out test split. Predictions are written to
    the experiment artifacts folder. When the test split has ground-truth annotations, detection
    metrics are computed via ``metric_calculations`` and logged through ``HafniaLogger``; if no
    ground truth is present the metric step is skipped with a warning.
    """
    # 1. Setup
    # Check cuda availability
    has_cuda = torch.cuda.is_available()
    if has_cuda:
        print("CUDA is available. Training on GPU.")
    else:
        print("CUDA is not available. Training on CPU.")

    # Define loggers
    logger = HafniaLogger(project_name=project_name)

    # Define data
    if hafnia_utils.is_hafnia_cloud_job():  # For hafnia cloud execution
        path_dataset = hafnia_utils.get_dataset_path_in_hafnia_cloud()  # The path to hidden dataset
        dataset = HafniaDataset.from_path(path_dataset)
    else:
        dataset = HafniaDataset.from_name("eccv-cross-city", version="latest")

    if samples is not None:
        dataset = dataset.select_samples(n_samples=samples)

    # Define pretrained weights
    checkpoints_folder_path = CKPT_DIR if CHECKPOINTS[inference_model_name].exists() else None
    checkpoint_model_path   = utils.get_checkpoint_if_available(logger, checkpoints_folder_path)
    if checkpoint_model_path is not None:
        user_logger.info(f"Using checkpoint '{checkpoint_model_path.name}' as pretrained model")
        model_path = checkpoint_model_path.as_posix()
        # Resuming from a checkpoint always uses its weights, regardless of the '--pretrained' flag.
        pretrained = True

    # Define model and trainer
    model_config    = InitModelConfig.load_model(model_path, use_weights=pretrained)
    model_primitive = model_config.task.primitive

    model_trainer = model_config.get_trainer()
    configuration = {
        "model": model_path,
        "pretrained": pretrained,
        "epochs": epochs,
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum_steps,
        "lr": lr,
        "resolution": resolution,
        "dataset": dataset.info.dataset_name,
        "has_cuda": has_cuda,
    }

    if has_cuda:
        configuration["device"]    = "cuda"
        configuration["num_gpus"]  = torch.cuda.device_count()
        configuration["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]

    logger.log_configuration(configuration)

    task_info = get_dataset_task_from_model_primitive(dataset, model_primitive, task_name)

    # Prepare Train/Val datasets
    dataset_train_val = dataset.create_split_dataset(split_name=[SplitName.TRAIN, SplitName.VAL])
    dataset_train_val = remove_images_with_no_bboxes(dataset_train_val, model_primitive=model_primitive)

    # Convert dataset to COCO format for training
    dataset_name = dataset_train_val.info.dataset_name
    dataset_path = Path(".data") / f"format_coco_roboflow_{dataset_name}"
    dataset_train_val.to_coco_format(dataset_path, task_name=task_info.name)
    path_experiment = logger._local_experiment_path
    path_experiment.mkdir(parents=True, exist_ok=True)

    # 2. Train
    if stop_early:
        user_logger.info("Early stopping before training was activated with '--stop_early' flag.")
        return None

    if run_train:
        model_trainer.train(
            dataset_dir=dataset_path.as_posix(),
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            grad_accum_steps=grad_accum_steps,
            output_dir=path_experiment.as_posix(),
            resolution=resolution,
            aug_config=AUG_CONFIG,
        )

        # 3. Save weights
        model_folder_path = logger.path_model()
        # Repackage each final checkpoint as a single compressed model archive in the model folder
        # (e.g. "checkpoint_best_regular.zip" and "checkpoint_best_total.zip").
        final_models = list(path_experiment.glob("checkpoint_*.pth"))
        model_path   = {}
        for checkpoint_path in final_models:
            model_name = checkpoint_path.stem  # e.g. "checkpoint_best_regular"
            model_checkpoint_path = model_folder_path / f"{model_name}.zip"
            model_config = InitModelConfig(name=model_config.name, task=task_info, model_weight_path=str(checkpoint_path))
            model_config.save_model(model_checkpoint_path)
            model_path[model_name] = model_checkpoint_path

        checkpoints_folder_path = logger.path_model_checkpoints()
        checkpoint_model_paths  = final_models  # For now we simply add final models as checkpoints
        for ckpt_path in checkpoint_model_paths:
            model_config = InitModelConfig(name=model_config.name, task=task_info, model_weight_path=str(ckpt_path))
            model_config.save_model(checkpoints_folder_path / f"{ckpt_path.stem}.zip")

    # 3. Test
    if run_test:
        if not run_train:
            # If training was not run in this execution, attempt to find a checkpoint
            # in the checkpoints folder to use for inference.
            if CHECKPOINTS[inference_model_name].exists():
                model_path = CHECKPOINTS

        # Prepare Test datasets
        # dataset_test = dataset.create_split_dataset(split_name=SplitName.VAL)  # For local testing only
        dataset_test = dataset.create_split_dataset(split_name=SplitName.TEST)
        inference_config = inference_config or InferenceConfig()
        inference_model  = WrappedModel.load_model(
            path_archive=model_path[inference_model_name],
            inference_config=inference_config,
            use_sahi=inference_sahi,
        )
        # inference_model.optimize_for_inference()  # Perform inside WrappedModel

        dataset_with_predictions = run_inference_on_dataset(dataset=dataset_test, model=inference_model)

        # Experiment output folder
        path_experiment_output_folder = logger._path_artifacts()
        # Save predictions to experiment output folder (drops unneeded columns)
        drop_columns = [SampleField.FILE_PATH, SampleField.VIDEO_INFO, SampleField.CAMERA_INFO, SampleField.META]
        dataset_with_predictions.samples = dataset_with_predictions.samples.drop(drop_columns, strict=False)
        dataset_with_predictions.write_annotations(path_experiment_output_folder)

        no_gt_data = dataset_test.samples.select(pl.col(task_info.primitive.column_name()).list.len()).sum().item() == 0
        if no_gt_data:  # Skip metric calculation for test sets without ground-truth annotations
            user_logger.warning("No ground-truth annotations found in the test set. Skipping metric calculation.")
        else:
            metrics = metric_calculations(prediction_dataset=dataset_with_predictions)
            for metric_name, metric_value in metrics.items():
                logger.log_metric(metric_name, metric_value, step=0)

    return logger


if __name__ == "__main__":
    # Creates launch schema file for the CLI function 'main'
    path_launch_schema = auto_save_command_builder_schema(main, cli_tool=utils.CLI_TOOL, order=0)
    user_logger.info(f"Launch schema saved to: {path_launch_schema}")
    app()

    # Pause for debugging
    time.sleep(60)

# endregion
