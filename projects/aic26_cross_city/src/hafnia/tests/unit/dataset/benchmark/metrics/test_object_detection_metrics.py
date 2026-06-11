import textwrap
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from hafnia.dataset.benchmark.benchmark import run_inference_on_dataset
from hafnia.dataset.dataset_names import TASK_NAME_PREDICTIONS_POSTFIX
from hafnia.dataset.hafnia_dataset import HafniaDataset
from hafnia.dataset.hafnia_dataset_types import DatasetInfo, Sample, TaskInfo
from hafnia.dataset.primitives import Bbox, Bitmask, Classification
from tests import helper_testing
from tests.helper_testing_benchmark import FakeInferenceModel


def test_calculate_mean_average_precision_perfect_predictions(tmp_path: Path):
    class_names = ["cat", "dog"]
    # run_inference_on_dataset reads each image from disk, so we materialize
    # small black JPEGs in a temporary directory rather than referencing
    # non-existent paths.
    samples = []
    for i in range(3):
        image_path = tmp_path / f"synthetic_img_{i}.jpg"
        Image.new("RGB", (500, 500), color=(0, 0, 0)).save(image_path)
        bboxes = [
            Bbox(top_left_x=0.0, top_left_y=0.0, width=0.4, height=0.4, class_name="cat"),
            Bbox(top_left_x=0.6, top_left_y=0.6, width=0.4, height=0.4, class_name="dog"),
        ]
        samples.append(Sample(file_path=str(image_path), split="train", height=500, width=500, bboxes=bboxes))

    info = DatasetInfo(
        dataset_name="test_map",
        tasks=[TaskInfo.from_class_names(primitive=Bbox, class_names=class_names)],
    )
    gt_dataset = HafniaDataset.from_samples_list(samples, info=info)

    model = FakeInferenceModel(fake_model_tasks=gt_dataset.info.tasks)
    model_task_names_before = [t.name for t in model.get_model_info().tasks]
    dataset_with_predictions = run_inference_on_dataset(dataset=gt_dataset, model=model)
    model_task_names_after = [t.name for t in model.get_model_info().tasks]
    assert model_task_names_before == model_task_names_after, (
        f"Model task names should not be modified by run_inference_on_dataset, "
        f"expected {model_task_names_before}, got {model_task_names_after}"
    )
    gt_task_name = Bbox.default_task_name()
    pred_task_name = f"{gt_task_name}{TASK_NAME_PREDICTIONS_POSTFIX}"
    metrics = dataset_with_predictions.calculate_mean_average_precision(
        task_name_predictions=pred_task_name,
        task_name_ground_truth=gt_task_name,
    )

    assert metrics.mAP == pytest.approx(1.0), f"Expected mAP=1.0, got {metrics.mAP}"
    assert metrics.mAP_50 == pytest.approx(1.0), f"Expected mAP_50=1.0, got {metrics.mAP_50}"
    assert metrics.AR_100 == pytest.approx(1.0), f"Expected AR_100=1.0, got {metrics.AR_100}"


def test_calculate_mean_average_precision_imperfect_predictions(tmp_path: Path):
    class_names = ["cat", "dog", "bird"]
    gt_task_name = Bbox.default_task_name()
    pred_task_name = f"{gt_task_name}{TASK_NAME_PREDICTIONS_POSTFIX}"

    def pred(**kwargs: Any) -> Bbox:
        return Bbox(ground_truth=False, task_name=pred_task_name, **kwargs)

    # Shifting a 0.4x0.4 box by 0.1 (normalized) yields IoU ≈ 0.6, so the shifted
    # predictions match at IoU=0.50 but fail at IoU=0.75.
    shift = 0.1
    samples = []
    for i in range(3):
        image_path = tmp_path / f"synthetic_img_{i}.jpg"
        Image.new("RGB", (500, 500), color=(0, 0, 0)).save(image_path)
        gt_bboxes = [
            Bbox(top_left_x=0.0, top_left_y=0.0, width=0.4, height=0.4, class_name="cat"),
            Bbox(top_left_x=0.6, top_left_y=0.6, width=0.4, height=0.4, class_name="dog"),
            Bbox(top_left_x=0.0, top_left_y=0.6, width=0.3, height=0.3, class_name="bird"),
            Bbox(top_left_x=0.6, top_left_y=0.0, width=0.3, height=0.3, class_name="bird"),
        ]
        pred_bboxes = [
            # Shifted cat prediction → IoU ≈ 0.6 (true positive at IoU=0.50, miss at IoU=0.75)
            pred(top_left_x=shift, top_left_y=0.0, width=0.4, height=0.4, class_name="cat", confidence=0.95),
            # Perfect dog prediction
            pred(top_left_x=0.6, top_left_y=0.6, width=0.4, height=0.4, class_name="dog", confidence=0.9),
            # Only one of the two bird GT boxes detected (false negative on the other)
            pred(top_left_x=0.0, top_left_y=0.6, width=0.3, height=0.3, class_name="bird", confidence=0.8),
            # Spurious low-confidence false positive in empty region
            pred(top_left_x=0.4, top_left_y=0.4, width=0.15, height=0.15, class_name="cat", confidence=0.3),
            # Misclassified dog: right location but wrong class
            pred(top_left_x=0.6, top_left_y=0.0, width=0.3, height=0.3, class_name="dog", confidence=0.5),
        ]
        samples.append(
            Sample(
                file_path=str(image_path),
                split="train",
                height=500,
                width=500,
                bboxes=gt_bboxes + pred_bboxes,
            )
        )

    info = DatasetInfo(
        dataset_name="test_map_imperfect",
        tasks=[
            TaskInfo.from_class_names(primitive=Bbox, class_names=class_names, name=gt_task_name),
            TaskInfo.from_class_names(primitive=Bbox, class_names=class_names, name=pred_task_name),
        ],
    )
    dataset = HafniaDataset.from_samples_list(samples, info=info)

    metrics = dataset.calculate_mean_average_precision(
        task_name_predictions=pred_task_name,
        task_name_ground_truth=gt_task_name,
    )

    # This test uses the metric report to make a readable and numerically stable assertion on metrics
    actual_report = metrics.report()
    print(actual_report)
    expected_report = textwrap.dedent("""\
Object Detection Metrics (COCO mAP)
===================================
  mAP    =  0.602   AP at IoU=0.50:0.05:0.95 (primary challenge metric)
  mAP_50 =  0.835   AP at IoU=0.50 (PASCAL VOC metric)
  mAP_75 =  0.502   AP at IoU=0.75 (strict metric)
  mAP_s  = -1.000   AP for small objects (area < 32² px)
  mAP_m  = -1.000   AP for medium objects (32² < area < 96² px)
  mAP_l  =  0.602   AP for large objects (area > 96² px)
  AR_1   =  0.600   AR given 1 detection per image
  AR_10  =  0.600   AR given 10 detections per image
  AR_100 =  0.600   AR given 100 detections per image
  AR_s   = -1.000   AR for small objects (area < 32² px)
  AR_m   = -1.000   AR for medium objects (32² < area < 96² px)
  AR_l   =  0.600   AR for large objects (area > 96² px)""")
    assert actual_report == expected_report


def test_calculate_mean_average_precision_invalid_tasks():
    """
    Tests that calculate_mean_average_precision raises informative errors when the prediction and ground-truth
    tasks are non-matching or invalid in various ways.
    """
    sample = Sample(file_path="some-path", split="train", height=100, width=100)
    gt_task_name = Bbox.default_task_name()
    pred_task_name = f"{gt_task_name}{TASK_NAME_PREDICTIONS_POSTFIX}"

    # Stage 1: ground-truth task uses an unsupported primitive (Classification).
    info = DatasetInfo(
        dataset_name="test_map_raises",
        tasks=[TaskInfo.from_class_names(primitive=Classification, class_names=["a"], name=gt_task_name)],
    )
    dataset = HafniaDataset.from_samples_list([sample], info=info)
    with pytest.raises(ValueError, match="Unsupported primitive type"):
        dataset.calculate_mean_average_precision(
            task_name_predictions=pred_task_name,
            task_name_ground_truth=gt_task_name,
        )

    # Stage 2: ground-truth primitive is fixed (Bbox), but GT task has no class names.
    info = DatasetInfo(
        dataset_name="test_map_raises",
        tasks=[TaskInfo(primitive=Bbox, classes=None, name=gt_task_name)],
    )
    dataset = HafniaDataset.from_samples_list([sample], info=info)
    with pytest.raises(ValueError, match=f"Ground-truth task '{gt_task_name}' does not define any class names"):
        dataset.calculate_mean_average_precision(
            task_name_predictions=pred_task_name,
            task_name_ground_truth=gt_task_name,
        )

    # Stage 3: GT has class names, but the prediction task has no class names.
    info = DatasetInfo(
        dataset_name="test_map_raises",
        tasks=[
            TaskInfo.from_class_names(primitive=Bbox, class_names=["cat"], name=gt_task_name),
            TaskInfo(primitive=Bbox, classes=None, name=pred_task_name),
        ],
    )
    dataset = HafniaDataset.from_samples_list([sample], info=info)
    with pytest.raises(ValueError, match=f"Prediction task '{pred_task_name}' does not define any class names"):
        dataset.calculate_mean_average_precision(
            task_name_predictions=pred_task_name,
            task_name_ground_truth=gt_task_name,
        )

    # Stage 4: both tasks have class names, but they do not match.
    info = DatasetInfo(
        dataset_name="test_map_raises",
        tasks=[
            TaskInfo.from_class_names(primitive=Bbox, class_names=["cat"], name=gt_task_name),
            TaskInfo.from_class_names(primitive=Bbox, class_names=["dog"], name=pred_task_name),
        ],
    )
    dataset = HafniaDataset.from_samples_list([sample], info=info)
    with pytest.raises(ValueError, match="do not match"):
        dataset.calculate_mean_average_precision(
            task_name_predictions=pred_task_name,
            task_name_ground_truth=gt_task_name,
        )


@pytest.mark.parametrize("dataset_name", helper_testing.MICRO_DATASETS)
def test_calculate_mean_average_precision(dataset_name: str):
    path_dataset = helper_testing.get_path_micro_hafnia_dataset(dataset_name=dataset_name, force_update=False)
    gt_dataset = HafniaDataset.from_path(path_dataset)

    coco_primitives = (Bbox, Bitmask)
    model_tasks = [t for t in gt_dataset.info.tasks if t.primitive in coco_primitives]
    assert len(model_tasks) > 0, (
        f"Expected at least one task with primitive in {coco_primitives} in dataset tasks, "
        f"found: {[t.primitive for t in gt_dataset.info.tasks]}"
    )

    for model_task in model_tasks:
        if model_task.name is None:
            raise ValueError(f"Model task names cannot be None, found in dataset '{dataset_name}'")
        model = FakeInferenceModel(fake_model_tasks=[model_task])
        dataset_with_pred = run_inference_on_dataset(dataset=gt_dataset, model=model)

        pred_task_name = f"{model_task.name}{TASK_NAME_PREDICTIONS_POSTFIX}"
        metrics = dataset_with_pred.calculate_mean_average_precision(
            task_name_predictions=pred_task_name,
            task_name_ground_truth=model_task.name,
        )
        assert metrics.mAP == pytest.approx(1.0), (
            f"Expected mAP=1.0 for task '{model_task.name}' on dataset '{dataset_name}', "
            f"got {metrics.mAP}. Ground Truth is directly used as predictions, so mAP should be perfect."
        )

        metrics.print_report()
