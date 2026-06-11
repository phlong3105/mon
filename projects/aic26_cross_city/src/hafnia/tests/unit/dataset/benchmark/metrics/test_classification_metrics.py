from typing import List, Union

import pytest

from hafnia.dataset.benchmark.metrics.classification_metrics import PerClassClassificationMetrics
from hafnia.dataset.hafnia_dataset import HafniaDataset
from hafnia.dataset.hafnia_dataset_types import DatasetInfo, Sample, TaskInfo
from hafnia.dataset.primitives import Classification


def _make_dataset(gt_labels: List[Union[None, str]], pred_labels: List[Union[None, str]], class_names: List[str]):
    """Helper: build a HafniaDataset with GT and prediction classifications."""
    gt_task_name = Classification.default_task_name()
    pred_task_name = "predictions"

    samples = []
    for gt_label, pred_label in zip(gt_labels, pred_labels):
        classifications = []
        if gt_label is not None:
            classifications.append(Classification(class_name=gt_label, task_name=gt_task_name, ground_truth=True))

        if pred_label is not None:
            classifications.append(
                Classification(class_name=pred_label, task_name=pred_task_name, confidence=0.9, ground_truth=False)
            )
        samples.append(
            Sample(file_path="/tmp/fake.jpg", split="test", height=100, width=100, classifications=classifications)
        )

    info = DatasetInfo(
        dataset_name="test_cls",
        tasks=[
            TaskInfo.from_class_names(primitive=Classification, class_names=class_names),
            TaskInfo.from_class_names(primitive=Classification, class_names=class_names, name=pred_task_name),
        ],
    )
    return HafniaDataset.from_samples_list(samples, info=info), pred_task_name


def test_perfect_predictions():
    """All predictions match ground truth -> accuracy=1, F1=1 for all classes."""
    class_names = ["cat", "dog", "man"]
    gt = ["cat", "dog", "man", "cat", "dog", "man"]
    pred = ["cat", "dog", "man", "cat", "dog", "man"]

    dataset, pred_task = _make_dataset(gt_labels=gt, pred_labels=pred, class_names=class_names)
    metrics = dataset.calculate_classification_metrics(
        task_name_predictions=pred_task,
        task_name_ground_truth=Classification.default_task_name(),
    )

    assert metrics.accuracy == pytest.approx(1.0)
    assert metrics.precision_macro == pytest.approx(1.0)
    assert metrics.recall_macro == pytest.approx(1.0)
    assert metrics.f1_macro == pytest.approx(1.0)
    assert metrics.precision_weighted == pytest.approx(1.0)
    assert metrics.recall_weighted == pytest.approx(1.0)
    assert metrics.f1_weighted == pytest.approx(1.0)

    expected = PerClassClassificationMetrics.from_confusion(TP=2, FP=0, FN=0, TN=4)
    for cls in class_names:
        assert metrics.per_class[cls] == expected

    # Smoke test the report formatting (exact content tested in other cases)
    metrics.print_report()

    # Smoke test per class
    for cls in class_names:
        print(f"Metrics for class '{cls}':")
        metrics.per_class[cls].print_report()

    print("Done")


def test_partial_correctness():
    """Mixed correct/incorrect predictions with known expected values."""
    class_names = ["cat", "dog"]
    # 3 samples: cat->cat (correct), cat->dog (wrong), dog->dog (correct)
    gt = ["cat", "cat", "dog"]
    pred = ["cat", "dog", "dog"]

    dataset, pred_task = _make_dataset(gt_labels=gt, pred_labels=pred, class_names=class_names)
    metrics = dataset.calculate_classification_metrics(
        task_name_predictions=pred_task,
        task_name_ground_truth=Classification.default_task_name(),
    )

    assert metrics.accuracy == pytest.approx(2.0 / 3.0)

    expected_class_cat = PerClassClassificationMetrics.from_confusion(TP=1, FP=0, FN=1, TN=1)
    expected_class_dog = PerClassClassificationMetrics.from_confusion(TP=1, FP=1, FN=0, TN=1)
    assert metrics.per_class["cat"] == expected_class_cat
    assert metrics.per_class["dog"] == expected_class_dog

    # Macro average: (1.0+0.5)/2=0.75 precision, (0.5+1.0)/2=0.75 recall
    assert metrics.precision_macro == pytest.approx(0.75)
    assert metrics.recall_macro == pytest.approx(0.75)
    assert metrics.f1_macro == pytest.approx((expected_class_cat.f1 + expected_class_dog.f1) / 2.0)


def test_single_class():
    """Edge case: only one class in the dataset."""
    class_names = ["only_class"]
    gt = ["only_class", "only_class", "only_class"]
    pred = ["only_class", "only_class", "only_class"]

    dataset, pred_task = _make_dataset(gt_labels=gt, pred_labels=pred, class_names=class_names)
    metrics = dataset.calculate_classification_metrics(
        task_name_predictions=pred_task,
        task_name_ground_truth=Classification.default_task_name(),
    )

    assert metrics.accuracy == pytest.approx(1.0)
    assert metrics.f1_macro == pytest.approx(1.0)
    expected = PerClassClassificationMetrics.from_confusion(TP=3, FP=0, FN=0, TN=0)
    assert metrics.per_class["only_class"] == expected


def test_missing_ground_truth_dropped_and_counted():
    """Samples without GT classification are dropped but counted in n_gt_missing."""
    gt_task_name = Classification.default_task_name()
    class_names = ["cat", "dog", "man"]

    # 8 samples total; indices 1 and 6 are missing GT and should be dropped.
    gt_labels = ["cat", None, "dog", "cat", "man", "dog", None, "man"]
    pred_labels = ["cat", "cat", "dog", "dog", "man", "dog", "man", "cat"]
    # Evaluated pairs (6 samples):
    #   cat->cat, dog->dog, cat->dog, man->man, dog->dog, man->cat

    dataset, pred_task = _make_dataset(gt_labels=gt_labels, pred_labels=pred_labels, class_names=class_names)
    metrics = dataset.calculate_classification_metrics(
        task_name_predictions=pred_task,
        task_name_ground_truth=gt_task_name,
    )

    assert metrics.n_gt_missing == 2
    assert metrics.n_pred_missing == 0

    # 4 correct out of 6 evaluated
    assert metrics.accuracy == pytest.approx(4.0 / 6.0)

    expected_class_cat = PerClassClassificationMetrics.from_confusion(TP=1, FP=1, FN=1, TN=3)
    expected_class_dog = PerClassClassificationMetrics.from_confusion(TP=2, FP=1, FN=0, TN=3)
    expected_class_man = PerClassClassificationMetrics.from_confusion(TP=1, FP=0, FN=1, TN=4)
    assert metrics.per_class["cat"] == expected_class_cat
    assert metrics.per_class["dog"] == expected_class_dog
    assert metrics.per_class["man"] == expected_class_man

    assert metrics.f1_macro == pytest.approx(
        (expected_class_cat.f1 + expected_class_dog.f1 + expected_class_man.f1) / 3.0
    )


def test_missing_predictions_dropped_and_counted():
    """Samples without prediction classification are dropped but counted in n_pred_missing."""
    gt_task_name = Classification.default_task_name()
    class_names = ["cat", "dog", "man"]

    # 8 samples total; indices 1 and 7 are missing prediction and should be dropped.
    gt_labels = ["cat", "dog", "dog", "cat", "man", "dog", "man", "cat"]
    pred_labels = ["cat", None, "dog", "dog", "man", "dog", "cat", None]
    # Evaluated pairs (6 samples):
    #   cat->cat, dog->dog, cat->dog, man->man, dog->dog, man->cat

    dataset, pred_task = _make_dataset(gt_labels=gt_labels, pred_labels=pred_labels, class_names=class_names)
    metrics = dataset.calculate_classification_metrics(
        task_name_predictions=pred_task,
        task_name_ground_truth=gt_task_name,
    )

    assert metrics.n_gt_missing == 0
    assert metrics.n_pred_missing == 2

    # Missing predictions are counted as incorrect (FN for their GT class), not dropped.
    assert metrics.accuracy == pytest.approx(4.0 / 8.0)

    expected_class_cat = PerClassClassificationMetrics.from_confusion(TP=1, FP=1, FN=2, TN=4)
    expected_class_dog = PerClassClassificationMetrics.from_confusion(TP=2, FP=1, FN=1, TN=4)
    expected_class_man = PerClassClassificationMetrics.from_confusion(TP=1, FP=0, FN=1, TN=6)
    assert metrics.per_class["cat"] == expected_class_cat
    assert metrics.per_class["dog"] == expected_class_dog
    assert metrics.per_class["man"] == expected_class_man

    assert metrics.f1_macro == pytest.approx(
        (expected_class_cat.f1 + expected_class_dog.f1 + expected_class_man.f1) / 3.0
    )


def test_missing_classifications_column_raises():
    """Dataset without classifications column should raise ValueError."""
    info = DatasetInfo(
        dataset_name="empty",
        tasks=[TaskInfo.from_class_names(primitive=Classification, class_names=["a"])],
    )
    samples = [
        Sample(file_path="/tmp/fake.jpg", split="test", height=100, width=100),
    ]
    dataset = HafniaDataset.from_samples_list(samples, info=info)
    # Drop the classifications column to simulate a dataset without it
    dataset.samples = dataset.samples.drop("classifications", strict=False)

    with pytest.raises(ValueError, match="classifications"):
        dataset.calculate_classification_metrics(
            task_name_predictions="pred",
            task_name_ground_truth=Classification.default_task_name(),
        )
