from typing import Callable, Dict, Optional, Union

from pydantic import BaseModel

from hafnia.dataset import primitives
from hafnia.dataset.dataset_names import TASK_NAME_PREDICTIONS_POSTFIX
from hafnia.dataset.hafnia_dataset import HafniaDataset
from hafnia.log import user_logger


class MetricsCalculator(BaseModel):
    def __call__(self, dataset: HafniaDataset) -> Dict[str, float]:
        return {}


class BboxMetricsCalculator(MetricsCalculator):
    task_ground_truth: str
    task_predictions: str

    def __call__(self, dataset: HafniaDataset) -> Dict[str, float]:
        metrics = dataset.calculate_mean_average_precision(
            task_name_predictions=self.task_predictions,
            task_name_ground_truth=self.task_ground_truth,
        )
        return metrics.as_dict()


class BitmaskMetricsCalculator(BboxMetricsCalculator):  # Same metrics as bbox, just different primitive type
    pass


class ClassificationMetricsCalculator(MetricsCalculator):
    name: str = "Classification"
    task_ground_truth: str
    task_predictions: str

    def __call__(self, dataset: HafniaDataset) -> Dict[str, float]:
        metrics = dataset.calculate_classification_metrics(
            task_name_predictions=self.task_predictions,
            task_name_ground_truth=self.task_ground_truth,
        )
        return metrics.as_dict()


def metric_calculations(
    prediction_dataset: HafniaDataset,
    metric_calculators: Optional[Dict[str, Union[MetricsCalculator, Callable]]] = None,
    prediction_task_name_postfix: str = TASK_NAME_PREDICTIONS_POSTFIX,
    separator: str = "/",
) -> dict[str, float]:
    # We derive and check metric groups before inference to avoid unnecessary computation for invalid configurations
    if metric_calculators is None:
        metric_calculators = auto_derive_metric_calculators(
            dataset=prediction_dataset,
            prediction_task_name_postfix=prediction_task_name_postfix,
        )

    metrics = {}
    for metric_calculator_name, metric_calculator in metric_calculators.items():
        metrics_group_no_prefix = metric_calculator(prediction_dataset)
        metrics_group = {f"{metric_calculator_name}{separator}{k}": v for k, v in metrics_group_no_prefix.items()}
        metrics.update(metrics_group)
    return metrics


def auto_derive_metric_calculators(
    dataset: HafniaDataset,
    prediction_task_name_postfix: str = TASK_NAME_PREDICTIONS_POSTFIX,
) -> Dict[str, Union[MetricsCalculator, Callable]]:
    prediction_tasks = [
        t for t in dataset.info.tasks if t.name is not None and t.name.endswith(prediction_task_name_postfix)
    ]
    metric_calculators: Dict[str, Union[MetricsCalculator, Callable]] = {}
    for prediction_task in prediction_tasks:
        primitive_type = prediction_task.primitive
        if prediction_task.name is None:
            continue
        task_name_ground_truth = prediction_task.name.removesuffix(prediction_task_name_postfix)

        matching_dataset_tasks = dataset.info.get_tasks_by_primitive(primitive=primitive_type)
        matching_dataset_tasks = [t for t in matching_dataset_tasks if t.name == task_name_ground_truth]
        if len(matching_dataset_tasks) == 0:
            user_logger.warning(
                f"No matching dataset task found for prediction task '{prediction_task.name}' "
                f"with primitive '{prediction_task.primitive.__name__}'. "
                "Metrics will not be calculated for this task."
            )
            continue
        if len(matching_dataset_tasks) > 1:
            user_logger.warning(
                f"Multiple matching dataset tasks found for prediction task '{prediction_task.name}' "
                f"with primitive '{prediction_task.primitive.__name__}'. "
                f"Matching dataset tasks: {[t.name for t in matching_dataset_tasks]}. "
                "Metrics will not be calculated for this task due to ambiguity."
            )
            continue

        metric_name = task_name_ground_truth  # Using task name as metric group name.
        if primitive_type == primitives.Bbox:
            metric_calculators[metric_name] = BboxMetricsCalculator(
                task_ground_truth=task_name_ground_truth,
                task_predictions=prediction_task.name,
            )
        elif primitive_type == primitives.Bitmask:
            metric_calculators[metric_name] = BitmaskMetricsCalculator(
                task_ground_truth=task_name_ground_truth,
                task_predictions=prediction_task.name,
            )
        elif primitive_type == primitives.Classification:
            metric_calculators[metric_name] = ClassificationMetricsCalculator(
                task_ground_truth=task_name_ground_truth,
                task_predictions=prediction_task.name,
            )
        else:
            user_logger.warning(
                f"Unsupported primitive '{primitive_type.__name__}' for prediction task '{prediction_task.name}'. "
                "Metrics will not be calculated for this task."
            )

    return metric_calculators
