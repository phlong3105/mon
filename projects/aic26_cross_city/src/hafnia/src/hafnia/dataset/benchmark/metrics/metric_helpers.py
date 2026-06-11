from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple, Type

from hafnia.dataset.primitives import Primitive

if TYPE_CHECKING:
    from hafnia.dataset.hafnia_dataset import HafniaDataset
    from hafnia.dataset.hafnia_dataset_types import TaskInfo


def validate_matching_tasks(
    dataset: "HafniaDataset",
    task_name_ground_truth: str,
    task_name_predictions: str,
    expected_primitives: Tuple[Type[Primitive], ...],
) -> Tuple["TaskInfo", "TaskInfo", List[str]]:
    """Resolve GT and prediction tasks and ensure their primitive and class names match.

    Returns ``(gt_task, pred_task, class_names)``.
    """

    def _check_primitive(task: "TaskInfo", task_name: str, label: str) -> None:
        if task.primitive not in expected_primitives:
            expected = ", ".join(p.__name__ for p in expected_primitives)
            raise ValueError(
                f"Unsupported primitive type for {label.lower()} task '{task_name}': "
                f"{task.primitive.__name__}. Expected one of: {expected}."
            )

    gt_task = dataset.info.get_task_by_name(task_name_ground_truth)
    _check_primitive(gt_task, task_name_ground_truth, "Ground-truth")
    gt_class_names = gt_task.get_class_names() or []
    if not gt_class_names:
        raise ValueError(f"Ground-truth task '{task_name_ground_truth}' does not define any class names.")

    pred_task = dataset.info.get_task_by_name(task_name_predictions)
    _check_primitive(pred_task, task_name_predictions, "Prediction")
    pred_class_names = pred_task.get_class_names() or []
    if not pred_class_names:
        raise ValueError(f"Prediction task '{task_name_predictions}' does not define any class names.")
    if gt_class_names != pred_class_names:
        raise ValueError(
            f"Class names for ground-truth task '{task_name_ground_truth}' and prediction task "
            f"'{task_name_predictions}' do not match. "
            f"Ground-truth class names: {gt_class_names}. Prediction class names: {pred_class_names}."
        )

    return gt_task, pred_task, gt_class_names
