# Benchmarking Models on a `HafniaDataset`

The `hafnia.dataset.benchmark` module runs a model over a `HafniaDataset`,
stores predictions as a new task on a copy of the dataset, and computes
metrics against the ground truth. The whole flow uses regular `HafniaDataset`
objects, so predictions, ground truth and metrics live side-by-side and can
be inspected, visualized or exported like any other dataset.

## The `InferenceModel` interface

Any model that implements `InferenceModel` can be benchmarked. The interface
has two methods:

```python
from hafnia.dataset.benchmark.inference_model import InferenceModel, ImageType
from hafnia.dataset.hafnia_dataset_types import ModelInfo, TaskInfo
from hafnia.dataset.primitives import Bbox, Primitive

class MyModel(InferenceModel):
    def get_model_info(self) -> ModelInfo:
        return ModelInfo(
            name="MyDetector",
            tasks=[TaskInfo.from_class_names(primitive=Bbox, class_names=[...])],
        )

    def predict(self, images, sample_dict=None) -> list[Primitive]:
        # Return primitives in hafnia format (normalized coords, class_name set,
        # ground_truth=False, confidence=<float>).
        ...
```

Predictions must be returned as **hafnia primitives** with `ground_truth=False`
and a `confidence` value, so they can be stored alongside the ground truth in
the same dataset.

## Running inference and computing metrics

The simplest path is `benchmark.run_benchmark`, which runs inference and
computes the appropriate metrics in one call:

```python
from hafnia.dataset.benchmark import benchmark
from hafnia.dataset.hafnia_dataset import HafniaDataset

dataset = HafniaDataset.from_name("coco-2017", version="1.0.0").select_samples(n_samples=10)
model   = MyModel()

metrics, dataset_predictions = benchmark.run_benchmark(dataset=dataset, model=model)
```

If you want to inspect the prediction dataset before scoring, split the call:

```python
dataset_predictions = benchmark.run_inference_on_dataset(dataset=dataset, model=model)

# Compute a specific metric directly on the dataset
map_metrics = dataset_predictions.calculate_mean_average_precision(
    task_name_ground_truth="object_detection",
    task_name_predictions="object_detection/predictions",
)
map_metrics.print_report()

# Or auto-derive all relevant metrics from the dataset's tasks
metrics = benchmark.metric_calculations(prediction_dataset=dataset_predictions)
```

Predictions are added as a new task with the suffix
`/predictions` (controlled by `task_name_prediction_postfix`). The ground
truth task is untouched, which makes side-by-side visualization easy.

## Visualizing predictions

Because predictions are stored as primitives on `Sample`, you can filter and
draw them like any annotation:

```python
from hafnia.dataset.hafnia_dataset_types import Sample
from PIL import Image

sample = Sample(**dataset_predictions[0])
sample.bboxes = [b for b in sample.bboxes if not b.ground_truth and b.confidence >= 0.2]
Image.fromarray(sample.draw_annotations()).save("predictions.png")
```

## Metrics

The metric layer auto-derives applicable metrics from the dataset's tasks
(for example, mAP for object detection). Implement
`MetricsCalculator.__call__(dataset) -> dict[str, float]` and pass it via
`metric_calculators=` to plug in custom metrics. See
[src/hafnia/dataset/benchmark/metrics_calculator.py](../src/hafnia/dataset/benchmark/metrics_calculator.py)
for the built-in calculators and the auto-derivation logic.

## See also

- [examples/example_benchmark.py](../examples/example_benchmark.py) — runnable walkthrough wrapping torchvision's SSDLite as an `InferenceModel` and benchmarking it on COCO.
- [docs/dataset.md](dataset.md) — the `HafniaDataset` format used for both inputs and outputs of benchmarking.
