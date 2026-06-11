from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple, Union

from hafnia.dataset.benchmark.inference_model import InferenceModel
from hafnia.dataset.benchmark.metrics_calculator import MetricsCalculator, metric_calculations
from hafnia.dataset.dataset_names import TASK_NAME_PREDICTIONS_POSTFIX
from hafnia.dataset.dataset_recipe.recipe_types import RecipeTransform
from hafnia.dataset.hafnia_dataset import HafniaDataset
from hafnia.dataset.hafnia_dataset_types import Sample
from hafnia.log import user_logger
from hafnia.utils import progress_bar


def run_benchmark(
    dataset: HafniaDataset,
    model: InferenceModel,
    task_name_prediction_postfix: str = TASK_NAME_PREDICTIONS_POSTFIX,
    metric_calculators: Optional[Dict[str, Union[MetricsCalculator, Callable]]] = None,
    recipe_transforms: Optional[List[RecipeTransform]] = None,
) -> Tuple[dict[str, float], HafniaDataset]:
    dataset_predictions = run_inference_on_dataset(
        dataset=dataset,
        model=model,
        task_name_prediction_postfix=task_name_prediction_postfix,
    )

    recipe_transforms = recipe_transforms or []
    for transform in recipe_transforms:
        dataset_predictions = transform.build(dataset_predictions)

    metrics = metric_calculations(
        prediction_dataset=dataset_predictions,
        metric_calculators=metric_calculators,
        prediction_task_name_postfix=task_name_prediction_postfix,
    )
    return metrics, dataset_predictions


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
