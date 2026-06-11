from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Annotated, Optional, Type
# from venv import logger

import polars as pl
from cyclopts import App, Parameter
from hafnia import utils as hafnia_utils
from hafnia.dataset.benchmark.benchmark import metric_calculations, run_inference_on_dataset
from hafnia.dataset.dataset_names import SampleField, SplitName
from hafnia.dataset.hafnia_dataset import HafniaDataset, Sample
from hafnia.dataset.hafnia_dataset_types import TaskInfo
from hafnia.dataset.primitives import Bbox, Primitive
from hafnia.experiment import HafniaLogger
from hafnia.experiment.command_builder import auto_save_command_builder_schema
from hafnia.log import user_logger
from ultralytics import YOLO
# from ultralytics import settings
# settings.update({"mlflow": False}) # Disable MLflow integration to avoid dependency issues, since we're manually logging metrics to HafniaLogger

import trainer_yolo
from trainer_yolo import utils
from trainer_yolo.wrapped_model import InferenceConfig, InitModelConfig, WrappedModel

# YOLO = utils.patch_to_support_experiment_tracker_with_hafnia(YOLO)

CLI_TOOL = "cyclopts"
app = App(name="train", help="Ultralytics YOLO Training for Hafnia object detection datasets")

MODEL_NAME_OPTIONS = [f"pretrained_models/{d.name}.zip" for d in trainer_yolo.wrapped_model.MODEL_OPTIONS]


def _hash_file_path(path_file: Path) -> str:
    return hashlib.md5(path_file.as_posix().encode("utf-8")).hexdigest()[:12]


def _map_to_ultralytics_split_name(split_name: str) -> str | None:
    mapped = SplitName.map_split_name(split_name, strict=False)
    if mapped == SplitName.TRAIN:
        return "train"
    if mapped == SplitName.VAL:
        return "val"
    if mapped == SplitName.TEST:
        return "test"
    return None


def _prepare_ultralytics_dataset(dataset: HafniaDataset, path_output: Path, task_name: str | None) -> Path:
    path_output.mkdir(parents=True, exist_ok=True)

    bbox_task = dataset.info.get_task_by_task_name_and_primitive(task_name=task_name, primitive=Bbox)
    class_names = bbox_task.get_class_names() or []
    if len(class_names) == 0:
        raise ValueError("No class names found for bbox task. A class list is required for YOLO training.")

    split_names = dataset.samples[SampleField.SPLIT].unique().to_list()
    exported_splits: set[str] = set()

    for split_name in split_names:
        print(f"Processing split '{split_name}'...")
        ul_split = _map_to_ultralytics_split_name(split_name)
        if ul_split is None:
            user_logger.warning(f"Skipping unsupported split '{split_name}'")
            continue

        split_samples = dataset.samples.filter(pl.col(SampleField.SPLIT) == split_name)
        path_images = path_output / "images" / ul_split
        path_labels = path_output / "labels" / ul_split
        path_images.mkdir(parents=True, exist_ok=True)
        path_labels.mkdir(parents=True, exist_ok=True)

        for idx, sample_dict in enumerate(split_samples.iter_rows(named=True)):
            sample = Sample(**sample_dict)
            if sample.file_path is None:
                continue

            path_src = Path(sample.file_path)
            if not path_src.exists():
                user_logger.warning(f"Image path not found, skipping: {path_src}")
                continue

            unique_name = f"{idx:08d}_{_hash_file_path(path_src)}{path_src.suffix}"
            path_dst_image = path_images / unique_name
            path_dst_label = path_labels / f"{Path(unique_name).stem}.txt"

            shutil.copy2(path_src, path_dst_image)

            bboxes = sample.bboxes or []
            yolo_lines = []
            for bbox in bboxes:
                class_idx = bbox.class_idx
                if class_idx is None:
                    if bbox.class_name is None:
                        continue
                    class_idx = class_names.index(bbox.class_name)

                x_center = bbox.top_left_x + bbox.width / 2
                y_center = bbox.top_left_y + bbox.height / 2
                yolo_lines.append(f"{class_idx} {x_center} {y_center} {bbox.width} {bbox.height}")

            path_dst_label.write_text("\n".join(yolo_lines))

        exported_splits.add(ul_split)

    if "train" not in exported_splits:
        raise ValueError("No train split was exported. A train split is required for training.")

    has_val = "val" in exported_splits
    val_target = "val" if has_val else ("test" if "test" in exported_splits else "train")

    path_data_yaml = path_output / "dataset.yaml"
    yaml_lines = [
        f"path: {path_output.as_posix()}",
        "train: images/train",
        f"val: images/{val_target}",
    ]
    if "test" in exported_splits:
        yaml_lines.append("test: images/test")

    yaml_lines.append("names:")
    for idx, name in enumerate(class_names):
        yaml_lines.append(f"  {idx}: {json.dumps(name)}")

    path_data_yaml.write_text("\n".join(yaml_lines) + "\n")
    return path_data_yaml


def get_dataset_task_from_model_primitive(
    dataset: HafniaDataset,
    model_primitive: Type[Primitive],
    task_name: Optional[str] = None,
) -> TaskInfo:
    matching_tasks = dataset.info.get_tasks_by_primitive(model_primitive)
    if len(matching_tasks) == 1:
        return matching_tasks[0]
    if len(matching_tasks) == 0:
        raise ValueError(f"Dataset requires '{model_primitive}' annotations.")
    if task_name is None:
        raise ValueError("Multiple tasks found. Specify '--task_name'.")
    model_task_info = dataset.info.get_task_by_name(task_name)
    if model_task_info.primitive != model_primitive:
        raise ValueError(f"Task '{task_name}' does not match primitive '{model_primitive}'.")
    return model_task_info


def save_launch_schema() -> Path | None:
    if hafnia_utils.is_hafnia_cloud_job():
        return None
    try:
        path_launch_schema = auto_save_command_builder_schema(main, cli_tool=CLI_TOOL, order=0)
        user_logger.info(f"Launch schema saved to: {path_launch_schema}")
        return path_launch_schema
    except Exception as exc:
        user_logger.warning(f"Schema generation failed: {exc}")
        return None


@app.default
def main(
    project_name: Annotated[str | None, Parameter(help="Project name for experiment tracking")] = "YOLO Detection",
    model_path: Annotated[
        str|None,
        Parameter(
            help=(
                "Path to a compressed (zip) pretrained model used as the training starting point. "
                f"Options: {MODEL_NAME_OPTIONS}"
            )
        ),
    ] = None,
    pretrained: Annotated[bool, Parameter(help="Initialize the model from pretrained weights")] = True,
    dataset_name: Annotated[str, Parameter(help="Local sample dataset name outside Hafnia")] = "midwest-vehicle-detection",
    dataset_version: Annotated[str, Parameter(help="Dataset version used for local runs")] = "1.0.1",
    task_name: Annotated[str | None, Parameter(help="Optional bbox task name")] = None,
    model: Annotated[str, Parameter(help="Ultralytics model name (e.g. yolo26l.pt) or path to .zip")] = "yolo26l.pt",
    epochs: Annotated[int, Parameter(help="Number of training epochs")] = 20,
    image_size: Annotated[int, Parameter(help="Training image size")] = 640,
    batch_size: Annotated[int, Parameter(help="Training batch size. Use -1 for auto")] = 16,
    learning_rate: Annotated[float, Parameter(help="Initial learning rate")] = 0.01,
    num_workers: Annotated[int, Parameter(help="Data loader workers")] = 8,
    patience: Annotated[int, Parameter(help="Early stopping patience")] = 30,
    device: Annotated[str, Parameter(help="Device string: cpu, 0, 0,1")] = "",
    seed: Annotated[int, Parameter(help="Random seed")] = 42,
    force_reexport: Annotated[bool, Parameter(help="Rebuild YOLO dataset")] = False,
    inference_model_name: Annotated[str, Parameter(help="Checkpoint used for post-training benchmark")] = "best",
    inference_config: Annotated[Optional[InferenceConfig], Parameter(help="Inference configuration")] = None,
):
    inference_config = inference_config or InferenceConfig()
    logger = HafniaLogger(project_name=project_name)

    if hafnia_utils.is_hafnia_cloud_job():
        path_dataset = hafnia_utils.get_dataset_path_in_hafnia_cloud()
        dataset = HafniaDataset.from_path(path_dataset)
    else:
        dataset = HafniaDataset.from_name(dataset_name, version=dataset_version)

    checkpoint_model_path = utils.get_checkpoint_if_available(logger)
    if checkpoint_model_path is not None:
        user_logger.info(f"Using checkpoint '{checkpoint_model_path.name}' as pretrained model")
        model_path = checkpoint_model_path.as_posix()
        # Resuming from a checkpoint always uses its weights, regardless of the '--pretrained' flag.
        pretrained = True

    print(f"Using model: {model}")

    model_config = InitModelConfig.load_model(model_path, use_weights=pretrained)
    # model_primitive = model_config.task.primitive

    path_export = Path(".data/tmp/ultralytics_dataset")
    if force_reexport and path_export.exists():
        shutil.rmtree(path_export)

    path_data_yaml = _prepare_ultralytics_dataset(dataset=dataset, path_output=path_export, task_name=task_name)

    if device == "":
        import torch
        device = "0" if torch.cuda.is_available() else "cpu"

    logger.log_configuration({
        "project_name": project_name,
        "model": model,
        "pretrained": pretrained,
        "epochs": epochs,
        "batch_size": batch_size,
        "image_size": image_size,
        "dataset": dataset.info.dataset_name,
        "device": device
    })

    task_info = get_dataset_task_from_model_primitive(dataset, Bbox, task_name)
    path_experiment = logger._local_experiment_path
    path_experiment.mkdir(parents=True, exist_ok=True)

    # Load YOLO Model and apply custom Hafnia MLflow callback
    yolo_model = model_config.get_trainer()
    # utils.patch_to_support_experiment_tracker_with_hafnia(yolo_model)

    train_kwargs = {
        "data": path_data_yaml.as_posix(),
        "epochs": epochs,
        "imgsz": image_size,
        "batch": batch_size,
        "lr0": learning_rate,
        "workers": num_workers,
        "patience": patience,
        "seed": seed,
        "project": path_experiment.as_posix(),
        "name": "yolo_train",
        "exist_ok": True,
        "device": device,
    }

    # Execute training loop
    yolo_model.train(**train_kwargs)

    path_save_dir = Path(yolo_model.trainer.save_dir)
    path_weights = path_save_dir / "weights"
    # path_best = path_weights / "best.pt"
    # path_last = path_weights / "last.pt"

    model_folder_path = logger.path_model()
    checkpoints_folder_path = logger.path_model_checkpoints()

    final_models = list(path_weights.glob("*.pt"))
    model_path = {}
    for checkpoint_path in final_models:
        model_name = checkpoint_path.stem  # e.g. "checkpoint_best_regular"
        model_checkpoint_path = model_folder_path / f"{model_name}.zip"
        model_config = InitModelConfig(name=model_config.name, task=task_info, model_weight_path=str(checkpoint_path))
        model_config.save_model(model_checkpoint_path)
        model_path[model_name] = model_checkpoint_path

    checkpoint_model_paths = final_models  # For now we simply add final models as checkpoints
    for ckpt_path in checkpoint_model_paths:
        model_config = InitModelConfig(name=model_config.name, task=task_info, model_weight_path=str(ckpt_path))
        model_config.save_model(checkpoints_folder_path / f"{ckpt_path.stem}.zip")

    # if path_best.exists():
    #     shutil.copy2(path_best, model_folder_path / "model.pt")
    #     shutil.copy2(path_best, checkpoints_folder_path / "best.pt")
    # if path_last.exists():
    #     shutil.copy2(path_last, checkpoints_folder_path / "last.pt")

    # # Save standard .zip packages expected by Hafnia platform
    # saved_model_paths = {}
    # for pt_name, ckpt_name in [("best.pt", "checkpoint_best"), ("last.pt", "checkpoint_last")]:
    #     path_pt = path_weights / pt_name
    #     if path_pt.exists():
    #         zip_name = f"{ckpt_name}.zip"
    #         cfg = InitModelConfig(name=model, task=task_info, model_weight_path=path_pt.as_posix())

    #         cfg.save_model(model_folder_path / zip_name)
    #         cfg.save_model(checkpoints_folder_path / zip_name)
    #         saved_model_paths[ckpt_name] = model_folder_path / zip_name

    # Post-Training Benchmark on TEST split
    # if inference_model_name in saved_model_paths:

    dataset_test = dataset.create_split_dataset(split_name=SplitName.TEST)
    inference_model = WrappedModel.load_model(model_path[inference_model_name], inference_config=inference_config)
    inference_model.optimize_for_inference()

    dataset_with_predictions = run_inference_on_dataset(dataset=dataset_test, model=inference_model)

    path_experiment_output_folder = logger._path_artifacts()
    drop_columns = [SampleField.FILE_PATH, SampleField.VIDEO_INFO, SampleField.CAMERA_INFO, SampleField.META]
    dataset_with_predictions.samples = dataset_with_predictions.samples.drop(drop_columns, strict=False)
    dataset_with_predictions.write_annotations(path_experiment_output_folder)

    no_gt_data = dataset_test.samples.select(pl.col(task_info.primitive.column_name()).list.len()).sum().item() == 0
    if no_gt_data:
        user_logger.warning("No ground-truth annotations found in test set. Skipping metrics.")
    else:
        metrics = metric_calculations(prediction_dataset=dataset_with_predictions)
        for metric_name, metric_value in metrics.items():
            logger.log_metric(metric_name, metric_value, step=0)
    # else:
    #     user_logger.warning(f"Could not benchmark. Checkpoint '{inference_model_name}' was not found.")

    return logger


if __name__ == "__main__":
    save_launch_schema()
    app()

    #pause for debugging
    import time
    time.sleep(600)
