from hafnia.dataset.benchmark.inference_model import InferenceModel
from hafnia.dataset.hafnia_dataset_types import ModelInfo, Sample, TaskInfo
from hafnia.dataset.primitives import Primitive


class FakeInferenceModel(InferenceModel):
    """A fake inference model that simply copies ground truth annotations to predictions with a confidence score."""

    def __init__(self, fake_model_tasks: list[TaskInfo]):
        self.fake_model_tasks = fake_model_tasks

    def predict(self, images, sample_dict=None) -> list[Primitive]:
        sample = Sample(**sample_dict)
        model_primitive_types = [t.primitive for t in self.fake_model_tasks]
        model_task_name = [t.name for t in self.fake_model_tasks]
        gt_annotations_all_tasks = sample.get_primitives(primitive_types=model_primitive_types)
        gt_annotations = [ann for ann in gt_annotations_all_tasks if ann.task_name in model_task_name]

        predictions = []
        for ann in gt_annotations:
            pred_ann: Primitive = ann.model_copy(deep=True)
            pred_ann.confidence = 0.9
            pred_ann.ground_truth = False
            predictions.append(pred_ann)
        return predictions

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(name="FakeModel", tasks=self.fake_model_tasks)
