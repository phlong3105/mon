import pytest

from hafnia.dataset.benchmark.benchmark import run_benchmark
from hafnia.dataset.hafnia_dataset import HafniaDataset
from tests import helper_testing
from tests.helper_testing_benchmark import FakeInferenceModel


@pytest.mark.parametrize("dataset_name", helper_testing.MICRO_DATASETS)
def test_benchmark(dataset_name: str):
    path_dataset = helper_testing.get_path_micro_hafnia_dataset(dataset_name=dataset_name, force_update=False)

    gt_dataset = HafniaDataset.from_path(path_dataset)
    model = FakeInferenceModel(fake_model_tasks=gt_dataset.info.tasks)
    task_name_prediction_postfix = "/some_predictions"
    metrics, dataset_predictions = run_benchmark(
        dataset=gt_dataset,
        model=model,
        task_name_prediction_postfix=task_name_prediction_postfix,
    )

    dataset_prediction_tasks = {t.name for t in dataset_predictions.info.tasks}
    assert len(metrics) > 0, "Expected at least one metric to be calculated"
    assert len(dataset_predictions) == len(gt_dataset)
    for model_task in model.get_model_info().tasks:
        assert model_task.name is not None, "Model task names cannot be None"
        prediction_task_name = model_task.name + task_name_prediction_postfix
        assert prediction_task_name in dataset_prediction_tasks
