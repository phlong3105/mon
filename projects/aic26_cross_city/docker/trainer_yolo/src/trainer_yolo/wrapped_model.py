import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple, Type, Union

import torch
from hafnia.dataset.benchmark.inference_model import ImageType, InferenceModel
from hafnia.dataset.hafnia_dataset_types import Bitmask, ModelInfo, TaskInfo
from hafnia.dataset.primitives import Bbox, Primitive
from hafnia.log import user_logger
from pydantic import BaseModel
from ultralytics import YOLO

MODEL_CONFIG_NAME = "model_config.json"


@dataclass
class ModelOption:
    name: str
    pretrained: bool
    supported: bool


MODEL_OPTIONS = [
    ModelOption(name="yolo26n", pretrained=True, supported=True),
    ModelOption(name="yolo26s", pretrained=True, supported=True),
    ModelOption(name="yolo26l", pretrained=True, supported=True),
    # ModelOption(name="yolo26x", pretrained=True, supported=False),
]
PATH_PRETRAINED_MODELS = Path(__file__).parent.parent.parent / "pretrained_models"


class InitModelConfig(BaseModel):
    name: str
    task: TaskInfo
    model_weight_path: Optional[str]

    def get_trainer(self):
        _, model_trainer = primitive_and_model_from_name(self.name, model_weights=self.model_weight_path)
        return model_trainer

    def save_model(self, path_archive: Union[str, Path]):
        """Save the model as a single compressed (zip) archive at ``path_archive``.

        The archive bundles the serialized model config (with a relative weight path) together
        with the weights file. Any existing archive at the destination is overwritten.
        """
        path_archive = Path(path_archive)
        path_archive.parent.mkdir(parents=True, exist_ok=True)

        # The config stores the weights as a relative filename so it resolves inside the archive.
        weight_name = None
        if self.model_weight_path is not None:
            weight_name = Path(self.model_weight_path).name
            # print(f"Saving model weights from {self.model_weight_path} as {weight_name} in the archive.")
        config_json = self.model_copy(update={"model_weight_path": weight_name}).model_dump_json(indent=4)
        # print(f"Model config JSON to be saved:\n{config_json}")

        with zipfile.ZipFile(path_archive, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(MODEL_CONFIG_NAME, config_json)
            if self.model_weight_path is not None:
                archive.write(self.model_weight_path, arcname=weight_name)

        # No compression and no archiving for simplicity, just save the config and weights directly to the specified paths.
        # config_json = self.model_dump_json(indent=4)
        # with open(path_archive.parent / MODEL_CONFIG_NAME, "w") as f:
        #     f.write(config_json)
        # if self.model_weight_path is not None:
        #     weight_dest = path_archive.parent / Path(self.model_weight_path).name
        #     torch.save(torch.load(self.model_weight_path), weight_dest)
        #     print(f"Saved model weights to {weight_dest}")

    @staticmethod
    def load_model(path_archive: Union[str, Path], use_weights: bool) -> "InitModelConfig":
        path_archive = Path(path_archive)
        # The weights are extracted to a temporary directory that persists for the lifetime of
        # the process, so they remain on disk when the trainer loads them via ``get_trainer``.
        extract_dir = Path(tempfile.mkdtemp(prefix="trainer_model_"))
        model_config: InitModelConfig = _load_config_and_weights(path_archive, extract_dir)

        if use_weights and model_config.model_weight_path is None:
            user_logger.warning(
                f"The specified model '{path_archive}' does not have pretrained weights available, but "
                "'pretrained=True' was set. The model will be trained from scratch."
            )

        if not use_weights and model_config.model_weight_path is not None:
            user_logger.warning(
                f"The specified model '{path_archive}' has pretrained weights available, but "
                "'pretrained=False' was set. The model will be trained from scratch without using the pretrained weights."
            )
        return model_config


class InferenceConfig(BaseModel):
    compile: bool = True
    batch_size: int = 1
    threshold: float = 0.05


class WrappedModel(InferenceModel):
    def __init__(self, model: YOLO, task: TaskInfo, inference_config: InferenceConfig):
        self.model = model
        self.task = task
        self.inference_config = inference_config

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(name=self.model.__class__.__name__, tasks=[self.task])

    def optimize_for_inference(self):
        # Ultralytics uses .fuse() to optimize models for inference (fuses Conv2d and BatchNorm2d)
        self.model.fuse()
        
        # If compilation is requested, torch.compile can be applied depending on PyTorch version
        # Note: YOLO has native handling for export formats like TensorRT, but for raw PyTorch:
        # if self.inference_config.compile and hasattr(torch, "compile"):
        #     self.model.model = torch.compile(self.model.model)

    def predict(self, images: Union[ImageType, List[ImageType]], sample_dict: Optional[dict] = None) -> List[Primitive]:
        # Ultralytics uses `conf` for the confidence threshold
        results = self.model.predict(
            source=images, 
            conf=self.inference_config.threshold, 
            verbose=False
        )
        
        bboxes = []
        for result in results:
            # result.orig_shape gives (height, width)
            image_shape = result.orig_shape 
            bboxes.extend(to_bbox_primitives(result, image_shape, bbox_task=self.task))
        
        return bboxes

    @staticmethod
    def load_model(path_archive: Union[str, Path], inference_config: InferenceConfig) -> "WrappedModel":
        path_archive = Path(path_archive)
        # Weights are extracted into a temporary directory and loaded into the model while the
        # directory is still alive; the extracted file is no longer needed once the model is built.
        with tempfile.TemporaryDirectory(prefix="trainer_model_") as extract_dir:
            model_config = _load_config_and_weights(path_archive, Path(extract_dir))
            primitive, model = primitive_and_model_from_name(
                model_config.name, model_weights=str(model_config.model_weight_path)
            )

        if primitive != model_config.task.primitive:
            raise ValueError(
                f"Model '{model_config.name}' is associated with primitive '{primitive.__name__}', "
                f"but the task in the config file requires primitive '{model_config.task.primitive.__name__}'."
            )

        return WrappedModel(model=model, task=model_config.task, inference_config=inference_config)


def _load_config_and_weights(path_archive: Path, extract_dir: Path) -> InitModelConfig:
    """Read the model config from a zipped model archive and extract its weights into ``extract_dir``.

    The returned config's ``model_weight_path`` is rewritten to the absolute path of the extracted
    weights file, or left as ``None`` when the archive contains no weights.
    """
    with zipfile.ZipFile(path_archive, "r") as archive:
        # print all contents for debugging
        print("Archive contents:", archive.namelist())
        model_config = InitModelConfig.model_validate_json(archive.read(MODEL_CONFIG_NAME))
        if model_config.model_weight_path is not None:
            weight_name = Path(model_config.model_weight_path).name
            archive.extract(weight_name, path=extract_dir)
            model_config.model_weight_path = (extract_dir / weight_name).as_posix()
    return model_config


def primitive_and_model_from_name(
    model_name: str, model_weights: Optional[str] = "pretrained"
) -> Tuple[Type[Primitive], YOLO]:

    if model_name in ["yolo26n", "yolo26s", "yolo26l"]:
        primitive = Bbox
    else:
        raise ValueError(f"Model {model_name} not recognized.")

    # In Ultralytics, weights are auto-downloaded if you provide a standard name (e.g., yolov8n.pt).
    # Assuming `yolo26n` maps to a local file or standard name `yolo26n.pt`. Adjust as needed.
    if model_weights == "pretrained" or model_weights is None:
        weight_path = f"{model_name}.pt" 
    else:
        weight_path = model_weights
        
    model = YOLO(weight_path)
    return primitive, model


def to_bbox_primitives(result, image_shape: Tuple[int, int], bbox_task: TaskInfo) -> list[Bbox]:
    predictions_bboxes = []
    
    # YOLO results store bounding boxes in the .boxes attribute
    if result.boxes is None or len(result.boxes) == 0:
        return predictions_bboxes
        
    boxes = result.boxes
    
    for bbox, class_idx, confidence in zip(boxes.xyxy, boxes.cls, boxes.conf, strict=True):
        # Extract item from tensors and move to CPU
        cls_id = int(class_idx.item())
        conf = float(confidence.item())
        bbox_coords = bbox.cpu().numpy()
        
        # Check against background class definition
        is_background_class = cls_id == len(bbox_task.classes)
        if is_background_class:
            continue
            
        bbox_obj = Bbox(
            height=(bbox_coords[3] - bbox_coords[1]) / image_shape[0], # y2 - y1 / height
            width=(bbox_coords[2] - bbox_coords[0]) / image_shape[1],  # x2 - x1 / width
            top_left_x=bbox_coords[0] / image_shape[1],                # x1 / width
            top_left_y=bbox_coords[1] / image_shape[0],                # y1 / height
            class_idx=cls_id,
            class_name=bbox_task.classes[cls_id].name,
            confidence=conf,
            ground_truth=False,
        )
        predictions_bboxes.append(bbox_obj)
        
    return predictions_bboxes