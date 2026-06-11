from pathlib import Path
from typing import List, Union

import numpy as np
import torch
from PIL import Image
from rich import print as rprint
from torchvision.models.detection import ssdlite

from hafnia.dataset.benchmark import benchmark
from hafnia.dataset.benchmark.inference_model import ImageType, InferenceModel
from hafnia.dataset.hafnia_dataset import HafniaDataset
from hafnia.dataset.hafnia_dataset_types import ModelInfo, Sample, TaskInfo
from hafnia.dataset.primitives import Bbox, Primitive

COCO_VERSION = "1.0.0"

# The benchmark feature runs an 'InferenceModel' over a 'HafniaDataset', stores the
# predictions as a new dataset, and computes metrics against the ground truth.
# To run it on your own model, implement the 'InferenceModel' interface - below is
# an example that wraps a torchvision object detection model.


class TorchvisionSSDLite(InferenceModel):
    """Wraps torchvision's SSDLite320 MobileNet V3 Large as a Hafnia 'InferenceModel'.

    The predict method converts raw model output (pixel xyxy boxes) to hafnia 'Bbox'
    primitives (normalized xywh) so they can be stored and evaluated by hafnia.
    """

    def __init__(self, box_score_thresh: float = 0.5):
        weights = ssdlite.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        self.model = ssdlite.ssdlite320_mobilenet_v3_large(weights=weights, score_thresh=box_score_thresh)
        self.model.eval()
        self.preprocess = weights.transforms()
        self.class_names_org: List[str] = list(weights.meta["categories"])

    def get_model_info(self) -> ModelInfo:
        # Converts self.class_names to TaskInfo. We convert any "N/A" class names to "unused_{i}"
        # to not have class name duplications.
        class_names = []
        for i_class, class_name in enumerate(self.class_names_org):
            if class_name in ("__background__", "N/A"):
                # class_name = f"unused_{i_class}"
                continue

            class_names.append(class_name)
        tasks = [TaskInfo.from_class_names(primitive=Bbox, class_names=class_names)]
        return ModelInfo(name="SSDLite320_MobileNet_V3_Large", tasks=tasks)

    def predict(
        self,
        images: Union[ImageType, List[ImageType]],
        sample_dict: Union[dict, List[dict], None] = None,
    ) -> List[Primitive]:
        # 'benchmark' calls predict with a single np.ndarray image (from Sample.read_image).
        assert isinstance(images, np.ndarray), "Expected a single np.ndarray image"
        image_np = images  # HxWxC, uint8, RGB

        # Torchvision expects CHW float tensors - the 'weights.transforms()' pipeline
        # handles normalization and resizing expected by the pretrained model.
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)
        batch = [self.preprocess(image_tensor)]

        with torch.no_grad():
            prediction = self.model(batch)[0]

        image_height, image_width = image_np.shape[:2]
        bboxes: List[Primitive] = []
        for box, label_idx, score in zip(
            prediction["boxes"].tolist(),
            prediction["labels"].tolist(),
            prediction["scores"].tolist(),
        ):
            xmin, ymin, xmax, ymax = box
            bboxes.append(
                Bbox(
                    top_left_x=xmin / image_width,
                    top_left_y=ymin / image_height,
                    width=(xmax - xmin) / image_width,
                    height=(ymax - ymin) / image_height,
                    class_name=self.class_names_org[label_idx],
                    confidence=float(score),
                    ground_truth=False,
                )
            )
        return bboxes


# 1. Load the COCO dataset sample
dataset = HafniaDataset.from_name("coco-2017", version=COCO_VERSION).select_samples(n_samples=3, seed=42)
dataset.print_basic_stats()

# 2. Instantiate the inference model. Threshold is set very low as we want all model predictions
# for metric calculations.
model = TorchvisionSSDLite(box_score_thresh=0.001)

# 3) Run model inference on the dataset
dataset_predictions = benchmark.run_inference_on_dataset(dataset=dataset, model=model)

# Predictions are added as a new task 'object_detection/predictions'
dataset.print_basic_stats()


# 4) Calculate specific metrics
map_metrics = dataset_predictions.calculate_mean_average_precision(
    task_name_ground_truth="object_detection",  # The original ground truth task in the dataset
    task_name_predictions="object_detection/predictions",
)
map_metrics.print_report()


# 5) Calculate all metrics - Relevant metric to be calculated are automatically derived.
metrics = benchmark.metric_calculations(prediction_dataset=dataset_predictions)

# 6) Or run everything in one go benchmark in a single go
metrics, dataset_predictions = benchmark.run_benchmark(dataset=dataset, model=model)

# 8) Inspect the resulting metrics
rprint(metrics)

# 9) Visualize ground truth + predictions for the first sample.
visualize_threshold = 0.2
path_tmp = Path(".data/tmp")
path_tmp.mkdir(parents=True, exist_ok=True)
sample = Sample(**dataset_predictions[0])
sample.bitmasks = None
sample.bboxes = [box for box in sample.bboxes if box.confidence >= visualize_threshold and not box.ground_truth]

image_with_annotations = sample.draw_annotations()
Image.fromarray(image_with_annotations).save(path_tmp / "benchmark_sample_with_predictions.png")
