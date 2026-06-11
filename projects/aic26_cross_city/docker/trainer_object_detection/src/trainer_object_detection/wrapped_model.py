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
from rfdetr import config, detr
from rfdetr.assets.model_weights import download_pretrain_weights

MODEL_CONFIG_NAME = "model_config.json"


@dataclass
class ModelOption:
    name: str
    pretrained: bool
    supported: bool


MODEL_OPTIONS = [
    ModelOption(name="RFDETRNano", pretrained=True, supported=True),
    # ModelOption(name="RFDETRSmall", pretrained=True, supported=True),
    ModelOption(name="RFDETRMedium", pretrained=True, supported=True),
    ModelOption(name="RFDETRLarge", pretrained=True, supported=True),
    ModelOption(name="RFDETRSegNano", pretrained=True, supported=True),
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
        config_json = self.model_copy(update={"model_weight_path": weight_name}).model_dump_json(indent=4)

        with zipfile.ZipFile(path_archive, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(MODEL_CONFIG_NAME, config_json)
            if self.model_weight_path is not None:
                archive.write(self.model_weight_path, arcname=weight_name)

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
    def __init__(self, model: detr.RFDETR, task: TaskInfo, inference_config: InferenceConfig):
        self.model = model
        self.task = task
        self.inference_config = inference_config

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(name=self.model.__class__.__name__, tasks=[self.task])

    def optimize_for_inference(self):
        self.model.optimize_for_inference(
            compile=self.inference_config.compile,
            batch_size=self.inference_config.batch_size,
            dtype=torch.float32,
        )

    def predict(self, images: Union[ImageType, List[ImageType]], sample_dict: Optional[dict] = None) -> List[Primitive]:
        predictions = self.model.predict(images, threshold=self.inference_config.threshold)
        bboxes: List[Bbox] = to_bbox_primitives(predictions, images.shape[:2], bbox_task=self.task)
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
        model_config = InitModelConfig.model_validate_json(archive.read(MODEL_CONFIG_NAME))
        if model_config.model_weight_path is not None:
            weight_name = Path(model_config.model_weight_path).name
            archive.extract(weight_name, path=extract_dir)
            model_config.model_weight_path = (extract_dir / weight_name).as_posix()
    return model_config


def primitive_and_model_from_name(
    model_name: str, model_weights: Optional[str] = "pretrained"
) -> Tuple[
    Type[Primitive],
    detr.RFDETR,
]:

    if model_name == "RFDETRNano":
        primitive = Bbox
        model_class = detr.RFDETRNano
        model_config: config.RFDETRBaseConfig = config.RFDETRNanoConfig()

    elif model_name == "RFDETRSmall":
        primitive = Bbox
        model_class = detr.RFDETRSmall
        model_config: config.RFDETRBaseConfig = config.RFDETRSmallConfig()

    elif model_name == "RFDETRMedium":
        primitive = Bbox
        model_class = detr.RFDETRMedium
        model_config: config.RFDETRBaseConfig = config.RFDETRMediumConfig()

    elif model_name == "RFDETRLarge":
        primitive = Bbox
        model_class = detr.RFDETRLarge
        model_config: config.RFDETRBaseConfig = config.RFDETRLargeConfig()

    elif model_name == "RFDETRSegNano":
        primitive = Bitmask
        model_class = detr.RFDETRSegNano
        model_config: config.RFDETRBaseConfig = config.RFDETRSegNanoConfig()
    else:
        raise ValueError(f"Model {model_name} not recognized.")

    kwargs: dict[str, Any] = {}

    if model_weights == "pretrained":
        if not Path(model_config.pretrain_weights).exists():
            download_pretrain_weights(model_config.pretrain_weights)
        kwargs["pretrain_weights"] = model_config.pretrain_weights
    else:
        kwargs["pretrain_weights"] = model_weights
    model = model_class(**kwargs)
    return primitive, model


def to_bbox_primitives(predictions, image_shape: Tuple[int, int], bbox_task: TaskInfo) -> list[Bbox]:
    predictions_bboxes = []
    for bbox, class_idx, confidence in zip(predictions.xyxy, predictions.class_id, predictions.confidence, strict=True):
        # Model creates n+1 class indices, where the last index is "no object" or "__background__" class
        is_background_class = class_idx.item() == len(bbox_task.classes)
        if is_background_class:
            continue
        bbox = Bbox(
            height=(bbox[3] - bbox[1]) / image_shape[0],
            width=(bbox[2] - bbox[0]) / image_shape[1],
            top_left_x=bbox[0] / image_shape[1],
            top_left_y=bbox[1] / image_shape[0],
            class_idx=int(class_idx),
            class_name=bbox_task.classes[int(class_idx)].name,
            confidence=float(confidence),
            ground_truth=False,
        )
        predictions_bboxes.append(bbox)
    return predictions_bboxes
