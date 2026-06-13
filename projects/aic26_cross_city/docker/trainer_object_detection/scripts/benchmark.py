#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Annotated

import polars as pl
from cyclopts import App, Parameter
from hafnia.dataset.benchmark.benchmark import (
    metric_calculations,
    run_inference_on_dataset,
)
from hafnia.dataset.dataset_names import SampleField, SplitName
from hafnia.dataset.hafnia_dataset import HafniaDataset, Optional
from hafnia.experiment import HafniaLogger
from hafnia.experiment.command_builder import auto_save_command_builder_schema
from hafnia.log import user_logger
from hafnia.utils import get_dataset_path_in_hafnia_cloud, is_hafnia_cloud_job

from trainer_object_detection import utils
from trainer_object_detection.wrapped_model import InferenceConfig, WrappedModel

app = App(name="benchmark", help="Benchmark")

CLASS_MAPPING_OPTIONS = [None, *utils.CLASS_MAPPINGS.keys()]

""" Benchmarking examples
# Example: Benchmark pretrained model (RFDETRNano) for a vehicle detection task.
# The tricky part for this benchmark is that RFDETRNano is pretrained on coco datasets labels while the
# dataset have different labels. To solve this we needs to remap both the dataset and model predictions to
# a common label space. In this example we remap both to a common "vehicle detection"
# label space, but other remapping strategies are also possible.
python scripts/benchmark.py --model-class-mapping COCO2OnlyVehicle


hafnia experiment create --recipe-id 8618234d-b4da-4aa9-bb3e-3be86bb50369 --trainer-path . --cmd "python scripts/benchmark.py --model-class-mapping COCO2OnlyVehicle"

"""


@app.default
def main(
    model_path: Annotated[
        str, Parameter(help="Path to the trained model archive (.zip)")
    ] = "./pretrained_models/RFDETRLarge.zip",
    inference: Annotated[Optional[InferenceConfig], Parameter(help="Inference configuration for the model")] = None,
    model_class_mapping: Annotated[
        Optional[str],
        Parameter(
            help=(
                "Class mapping applied to the model predictions to remap them into a common label space "
                f"with the ground truth. Options: {CLASS_MAPPING_OPTIONS}"
            )
        ),
    ] = None,
    dataset_class_mapping: Annotated[
        Optional[str],
        Parameter(
            help=(
                "Class mapping applied to the dataset ground-truth labels to remap them into a common "
                f"label space with the predictions. Options: {CLASS_MAPPING_OPTIONS}"
            )
        ),
    ] = None,
    split_name: Annotated[str, Parameter(help="Dataset split to run on")] = SplitName.TEST,
    save_annotations: Annotated[
        bool,
        Parameter(
            help="Write the predictions (annotations only, no image data) to the experiment artifacts folder."
        ),
    ] = True,
    samples: Annotated[
        Optional[int],
        Parameter(help="Limit the number of samples to run on. Useful for faster testing."),
    ] = None,
):
    """Run a model on a Hafnia dataset split and compute detection metrics when ground truth is available.

    Loads the dataset (the hidden dataset when running on the Hafnia platform, otherwise a public
    sample dataset), runs the model on the requested split, and - when the split has ground-truth
    annotations - computes detection metrics and logs them through ``HafniaLogger``. When the split
    has no ground truth (e.g. a held-out test set without labels) the metric step is skipped, so the
    same script can also be used as a pure inference pass.

    The ``model_class_mapping`` and ``dataset_class_mapping`` flags project predictions and/or
    ground truth into a common label space, which is needed when a pretrained model (e.g. trained on
    COCO) is benchmarked against a dataset with a different label space. When ``save_annotations`` is
    set (default), the dataset with predictions appended as a new prediction task on each sample is
    written - annotations only, no image data - to the experiment artifacts folder for downstream
    analysis or visualization.
    """
    inference = inference or InferenceConfig()
    logger = HafniaLogger(project_name="Benchmarking RF-DETR")
    if is_hafnia_cloud_job():  # For hafnia cloud execution
        path_dataset = get_dataset_path_in_hafnia_cloud()  # The path to the full/hidden dataset is returned
        dataset = HafniaDataset.from_path(path_dataset)
    else:
        # The small/public sample dataset is returned by name
        dataset = HafniaDataset.from_name("midwest-vehicle-detection", version="1.0.0")

    # Prefer a user-selected checkpoint over the configured model when one is available.
    checkpoint_model_path = utils.get_checkpoint_if_available(logger)
    if checkpoint_model_path is not None:
        user_logger.info(f"Using checkpoint '{checkpoint_model_path.name}' instead of '{model_path}'")
        model_path = checkpoint_model_path.as_posix()

    model = WrappedModel.load_model(model_path, inference_config=inference)
    model.optimize_for_inference()

    dataset_split = dataset.create_split_dataset(split_name=split_name)
    dataset_task_info = dataset.info.get_task_by_primitive(model.task.primitive)

    configuration = {
        "model": model.__class__.__name__,
        "compile": inference.compile,
        "batch_size": inference.batch_size,
        "threshold": inference.threshold,
        "dataset": dataset.info.dataset_name,
        "dataset_version": dataset.info.version,
        "model_filename": Path(model_path).name,
        "num_samples": len(dataset_split),
        "class_mapping_model": model_class_mapping,
        "class_mapping_dataset": dataset_class_mapping,
        "split_name": split_name,
    }
    logger.log_configuration(configuration)

    if samples is not None:
        dataset_split = dataset_split.select_samples(n_samples=samples, seed=42)

    # Remap ground-truth classes to a common label space before inference if requested
    if dataset_class_mapping is not None:
        dataset_split = dataset_split.class_mapper(
            class_mapping=utils.CLASS_MAPPINGS[dataset_class_mapping],
            method="remove_undefined",
            task_name=model.task.name,
        )

    # Run inference on the dataset. Predictions are appended as new tasks on each sample.
    prediction_post_fix = "/predictions"
    dataset_with_predictions = run_inference_on_dataset(
        dataset=dataset_split,
        model=model,
        task_name_prediction_postfix=prediction_post_fix,
    )

    # Remap model prediction classes into the dataset's label space if requested
    if model_class_mapping is not None:
        dataset_with_predictions = dataset_with_predictions.class_mapper(
            class_mapping=utils.CLASS_MAPPINGS[model_class_mapping],
            method="remove_undefined",
            task_name=f"{dataset_task_info.name}{prediction_post_fix}",
        )

    # Save predictions to the experiment artifacts folder (annotations only, drops image-related columns)
    if save_annotations:
        drop_columns = [SampleField.FILE_PATH, SampleField.VIDEO_INFO, SampleField.CAMERA_INFO, SampleField.META]
        dataset_with_predictions.samples = dataset_with_predictions.samples.drop(drop_columns, strict=False)
        dataset_with_predictions.write_annotations(logger._path_artifacts())

    # Skip metric calculation for splits without ground-truth annotations
    gt_column = dataset_task_info.primitive.column_name()
    no_gt_data = dataset_split.samples.select(pl.col(gt_column).list.len()).sum().item() == 0
    if no_gt_data:
        user_logger.warning("No ground-truth annotations found in the selected split. Skipping metric calculation.")
        return logger

    metrics = metric_calculations(
        prediction_dataset=dataset_with_predictions,
        prediction_task_name_postfix=prediction_post_fix,
    )
    for metric_name, metric_value in metrics.items():
        logger.log_metric(metric_name, metric_value, step=0)

    return logger


if __name__ == "__main__":
    # Creates launch schema file for the CLI function 'main'
    path_launch_schema = auto_save_command_builder_schema(main, cli_tool=utils.CLI_TOOL)
    user_logger.info(f"Launch schema saved to: {path_launch_schema}")
    app()
