#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Type

import numpy as np
import supervision as sv
import torch
import torchvision.transforms.functional as F
from ensemble_boxes import weighted_boxes_fusion
from hafnia.dataset.benchmark.inference_model import ImageType, InferenceModel
from hafnia.dataset.hafnia_dataset_types import Bitmask, ModelInfo, TaskInfo
from hafnia.dataset.primitives import Bbox, Primitive
from hafnia.log import user_logger
from pydantic import BaseModel
from rfdetr import config, detr
from rfdetr.assets.model_weights import download_pretrain_weights
from rfdetr.config import RFDETRBaseConfig
from rfdetr_plus.models import (
    RFDETR2XLarge,
    RFDETR2XLargeConfig,
    RFDETRXLarge,
    RFDETRXLargeConfig,
)


# ==============================================================================
# region CONSTANTS
# ==============================================================================

MODEL_CONFIG_NAME = "model_config.json"

@dataclass
class ModelOption:
    name: str
    pretrained: bool
    supported: bool


MODEL_OPTIONS = [
    ModelOption(name="RFDETRNano",    pretrained=True, supported=True),
    ModelOption(name="RFDETRSmall",   pretrained=True, supported=True),
    ModelOption(name="RFDETRMedium",  pretrained=True, supported=True),
    ModelOption(name="RFDETRLarge",   pretrained=True, supported=True),
    ModelOption(name="RFDETRSegNano", pretrained=True, supported=True),
]
PATH_PRETRAINED_MODELS = Path(__file__).parent.parent.parent / "pretrained_models"

# endregion


# ==============================================================================
# region BASE CLASSES
# ==============================================================================

class InitModelConfig(BaseModel):

    name: str
    task: TaskInfo
    model_weight_path: str

    def get_trainer(self):
        _, model_trainer, _ = primitive_and_model_from_name(self.name, model_weights=self.model_weight_path)
        return model_trainer

    def save_model(self, path_archive: str | Path):
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
    def load_model(path_archive: str | Path, use_weights: bool) -> "InitModelConfig":
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
    threshold: float = 0.05        # Note: threshold = 0.01 -> file too large, error

    ensemble: bool = False
    wbf_iou_thr: float = 0.55      # Overlap threshold for grouping boxes
    wbf_skip_box_thr: float = 0.0001  # Exclude boxes below this confidence prior to fusion

    sahi: bool = False
    ftta: bool = False
    ftta_steps: int = 3                 # Number of inner-loop optimization steps
    ftta_lr: float = 1e-4          # Small learning rate for test-time adaptation
    ftta_thr: float = 0.4          # Confidence cutoff to filter pseudo-labels

# endregion


# ==============================================================================
# region CONCRETE IMPLEMENTATIONS
# ==============================================================================

class WrappedModel(InferenceModel):

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        model: detr.RFDETR,
        task: TaskInfo,
        model_config: RFDETRBaseConfig,
        inference_config: InferenceConfig,
    ):
        self.model = model
        self.task = task
        self.model_config = model_config
        self.inference_config = inference_config
        self.slicer: sv.InferenceSlicer = None

        self.optimize_for_inference()
        if self.inference_config.sahi:
            self.optimize_for_sahi()

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(name=self.model.__class__.__name__, tasks=[self.task])

    def optimize_for_inference(self):
        self.model.optimize_for_inference(
            compile=self.inference_config.compile,
            batch_size=self.inference_config.batch_size,
            dtype=torch.float32,
        )

    def optimize_for_sahi(self):
        # 1. Define the callback for the Inference Slicer
        # This function tells supervision how to run your specific model on a single grid patch
        def slice_callback(image_patch: np.ndarray) -> sv.Detections:
            # Get predictions for the single patch
            detections = self.model.predict(
                image_patch,
                threshold=self.inference_config.threshold
            )
            detections.metadata = {}
            return detections

        # 2. Initialize the Native Supervision Slicer
        resolution = self.model_config.resolution
        self.slicer = sv.InferenceSlicer(
            callback=slice_callback,
            slice_wh=resolution,
            overlap_wh=(100, 100),  # 20% overlap between slices
            overlap_filter=sv.OverlapFilter.NON_MAX_SUPPRESSION,
            iou_threshold=0.1,
        )

    # --- Callable & Context Manager ---
    def predict(
        self,
        images: ImageType | list[ImageType],
        sample_dict: dict = None
    ) -> list[Primitive]:
        # Handle single images vs list of images uniformly
        image_list = images if isinstance(images, list) else [images]
        all_bboxes: list[Bbox] = []

        # Inference flags
        use_ftta = self.inference_config.ftta
        use_sahi = self.inference_config.sahi

        # Process each image
        for image in image_list:
            if use_ftta:
                # To perform isolated test-time adaptation per sample (preventing confirmation error explosion),
                # we checkpoint the original model state and restore it after processing the image.
                backup_state = copy.deepcopy(self.model.model.model.state_dict())
                self._run_ftta(image)

            # Inference with SAHI
            if use_sahi and self.slicer is not None:
                predictions = self.slicer(image)
                bboxes = to_bbox_primitives(predictions, image.shape[:2], bbox_task=self.task)
            # Normal inference without SAHI
            else:
                threshold = self.inference_config.threshold
                predictions = self.model.predict(images=image, threshold=threshold)
                bboxes = to_bbox_primitives(predictions, image.shape[:2], bbox_task=self.task)

            all_bboxes.extend(bboxes)

            if use_ftta:
                # Rollback to base parameters to avoid progressive drift on subsequent unrelated frames
                self.model.model.model.load_state_dict(backup_state)

        return all_bboxes

    def _run_ftta(self, image: ImageType):
        """Performs an unsupervised inner-loop calibration on a single test image
        using high-confidence predictions as pseudo-labels.
        """
        ftta_steps = self.inference_config.ftta_steps
        ftta_lr = self.inference_config.ftta_lr
        ftta_thr = self.inference_config.ftta_thr

        # Unfreeze only the parameters crucial for domain adaptation in a DETR structure
        # (e.g., Norm layers, transformer decoder query projection, and detection heads)
        trainable_params = []
        for name, param in self.model.model.model.named_parameters():
            if any(k in name for k in ["norm", "head", "query_embed", "decoder.layers"]):
                param.requires_grad = True
                trainable_params.append(param)
            else:
                param.requires_grad = False
        if not trainable_params:
            return

        # Use an optimizer with a small learning rate and high momentum to stabilize 1-sample updates
        optimizer = torch.optim.SGD(trainable_params, lr=ftta_lr, momentum=0.9)

        # Turn model to train mode to unlock dynamic computation graphs and track gradients
        self.model.model.model.train()

        # Inner loop
        for i in range(ftta_steps):
            optimizer.zero_grad()

            # Prepare input
            img_tensor = F.to_tensor(image)
            img_tensor = img_tensor.to(self.model.model.device)
            resize_to  = [self.model.model.resolution, self.model.model.resolution]
            img_tensor = F.resize(img_tensor, resize_to)
            img_tensor = F.normalize(img_tensor, self.model.means, self.model.stds)
            batch_tensor = torch.stack([img_tensor])

            # Forward pass: extract raw query outputs from RF-DETR
            # logits shape: [1, num_queries, num_classes + 1]
            # boxes shape:  [1, num_queries, 4]
            raw_outputs = self.model.model.model.forward(batch_tensor)

            pred_logits = raw_outputs["pred_logits"]
            pred_boxes  = raw_outputs["pred_boxes"]

            # Compute classification probabilities across all queries
            probs = torch.softmax(pred_logits, dim=-1)

            # A. Classification Loss: Minimizing Shannon Entropy forces the model to resolve
            # domain-shifted ambiguities and make confident foreground/background decisions.
            # We add a small epsilon to avoid NaN issues with log.
            entropy  = -torch.sum(probs * torch.log(probs + 1e-6), dim=-1)
            loss_cls = entropy.mean()

            # B. Geometric/Regression Loss: Filter queries where the model already has high confidence.
            # We enforce consistency on those reliable coordinates to anchor the spatial representations.
            with torch.no_grad():
                max_probs, _ = torch.max(probs[..., :-1], dim=-1)  # Exclude the background class channel
                confident_mask = max_probs > ftta_thr  # Shape: [1, num_queries]

            if confident_mask.sum() > 0:
                # Target stable bounding boxes from a detached pseudo-ground-truth reference
                target_boxes = pred_boxes.detach()
                # Compute Smooth L1/Huber loss over the localized query coordinates
                loss_bbox = torch.nn.functional.huber_loss(
                    pred_boxes[confident_mask],
                    target_boxes[confident_mask],
                    delta=1.0
                )
                # Combine losses (balance coefficient 2.0 keeps box stabilization matching entropy gradients)
                loss = loss_cls + 2.0 * loss_bbox
            else:
                loss = loss_cls

            # Parameter Update step
            loss.backward()
            optimizer.step()

        # Re-verify inference configuration defaults
        self.model.model.model.eval()

    # --- Creation ---
    @staticmethod
    def load_model(
        path_archive: str | Path,
        inference_config: InferenceConfig
    ) -> "WrappedModel":
        path_archive = Path(path_archive)
        # Weights are extracted into a temporary directory and loaded into the model while the
        # directory is still alive; the extracted file is no longer needed once the model is built.
        with tempfile.TemporaryDirectory(prefix="trainer_model_") as extract_dir:
            model_config = _load_config_and_weights(path_archive, Path(extract_dir))
            primitive, model, model_config_ = primitive_and_model_from_name(
                model_name=model_config.name,
                model_weights=str(model_config.model_weight_path)
            )

        if primitive != model_config.task.primitive:
            raise ValueError(
                f"Model '{model_config.name}' is associated with primitive '{primitive.__name__}', "
                f"but the task in the config file requires primitive '{model_config.task.primitive.__name__}'."
            )

        return WrappedModel(
            model=model,
            task=model_config.task,
            model_config=model_config_,
            inference_config=inference_config,
        )


class WrappedEnsembleModel(InferenceModel):

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        models: list[detr.RFDETR],
        task: TaskInfo,
        models_config: list[RFDETRBaseConfig],
        inference_config: InferenceConfig,
        model_weights: list[float] | None = None,
    ):
        self.models = models
        self.task = task
        self.models_config = models_config
        self.inference_config = inference_config

        # If no specific weights are passed, give all models uniform influence
        self.model_weights = model_weights if model_weights is not None else [1.0] * len(models)

        self.optimize_for_inference()

    def get_model_info(self) -> ModelInfo:
        model_names = ""
        for model in self.models:
            model_names += model.__class__.__name__ + "_"
        return ModelInfo(name=model_names, tasks=[self.task])

    def optimize_for_inference(self):
        for i in range(len(self.models)):
            self.models[i].optimize_for_inference(
                compile=self.inference_config.compile,
                batch_size=self.inference_config.batch_size,
                dtype=torch.float32,
            )

    # --- Callable & Context Manager ---
    def predict(
        self,
        images: ImageType | list[ImageType],
        sample_dict: dict = None
    ) -> list[Primitive]:
        # Handle single images vs. a list of images uniformly
        image_list = images if isinstance(images, list) else [images]
        all_bboxes: list[Bbox] = []

        # Inference configs
        threshold = self.inference_config.threshold
        wbf_iou_thr = self.inference_config.wbf_iou_thr
        wbf_skip_box_thr = self.inference_config.wbf_skip_box_thr

        # Process each image
        for image in image_list:
            # Form structures required by ensemble_boxes package
            boxes_per_model  = []
            scores_per_model = []
            labels_per_model = []

            # Gather predictions from each model for this specific image
            for i, model in enumerate(self.models):
                predictions = model.predict(images=image, threshold=threshold)
                bboxes = to_bbox_primitives(predictions, image.shape[:2], bbox_task=self.task)

                model_boxes  = []
                model_scores = []
                model_labels = []

                for bbox in bboxes:
                    # Convert your primitive format [top_left_x, top_left_y, width, height]
                    # into WBF expected normalized format: [x1, y1, x2, y2]
                    x1 = bbox.top_left_x
                    y1 = bbox.top_left_y
                    x2 = x1 + bbox.width
                    y2 = y1 + bbox.height
                    # Bound between [0.0, 1.0] to safeguard against rounding edge cases
                    norm_box = [
                        max(0.0, min(1.0, float(x1))),
                        max(0.0, min(1.0, float(y1))),
                        max(0.0, min(1.0, float(x2))),
                        max(0.0, min(1.0, float(y2)))
                    ]
                    model_boxes.append(norm_box)
                    model_scores.append(float(bbox.confidence))
                    model_labels.append(int(bbox.class_idx))

                boxes_per_model.append(model_boxes)
                scores_per_model.append(model_scores)
                labels_per_model.append(model_labels)

            # Apply Weighted Boxes Fusion if boxes were found by any of the ensemble members
            if any(len(b) > 0 for b in boxes_per_model):
                fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
                    boxes_list=boxes_per_model,
                    scores_list=scores_per_model,
                    labels_list=labels_per_model,
                    weights=self.model_weights,
                    iou_thr=wbf_iou_thr,
                    skip_box_thr=wbf_skip_box_thr
                )

                # Reconstruct fused coordinates directly back into your custom Bbox primitives
                for box, score, label_idx in zip(fused_boxes, fused_scores, fused_labels):
                    # Filter out any fused targets dropping below operational criteria
                    if score < threshold:
                        continue

                    # Convert WBF [x1, y1, x2, y2] format back into your [x, y, w, h] schema
                    x1, y1, x2, y2 = box
                    fused_w = x2 - x1
                    fused_h = y2 - y1

                    # Build your exact native target structure
                    fused_primitive = Bbox(
                        height=float(fused_h),
                        width=float(fused_w),
                        top_left_x=float(x1),
                        top_left_y=float(y1),
                        class_idx=int(label_idx),
                        class_name=self.task.classes[int(label_idx)].name,
                        confidence=float(score),
                        ground_truth=False,
                    )
                    all_bboxes.append(fused_primitive)

        return all_bboxes

    # --- Creation ---
    @staticmethod
    def load_model(
        path_archive: dict[str, str | Path],
        inference_config: InferenceConfig,
        model_weights: list[float] | None = None,
    ) -> "WrappedEnsembleModel":
        models: list[detr.RFDETR] = []
        models_config: list[RFDETRBaseConfig] = []

        for model_name, model_path in path_archive.items():
            path_archive = Path(model_path)
            # Weights are extracted into a temporary directory and loaded into the model while the
            # directory is still alive; the extracted file is no longer needed once the model is built.
            with tempfile.TemporaryDirectory(prefix="trainer_model_") as extract_dir:
                model_config = _load_config_and_weights(path_archive, Path(extract_dir))
                primitive, model, model_config_ = primitive_and_model_from_name(
                    model_name=model_config.name,
                    model_weights=str(model_config.model_weight_path)
                )

            if primitive != model_config.task.primitive:
                raise ValueError(
                    f"Model '{model_config.name}' is associated with primitive '{primitive.__name__}', "
                    f"but the task in the config file requires primitive '{model_config.task.primitive.__name__}'."
                )

            models.append(model)
            models_config.append(model_config_)

        return WrappedEnsembleModel(
            models=models,
            task=model_config.task,
            models_config=models_config,
            inference_config=inference_config,
            model_weights=model_weights,
        )

# endregion


# ==============================================================================
# region UTILS
# ==============================================================================

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
    model_name: str,
    model_weights: str = "pretrained"
) -> tuple[Type[Primitive], detr.RFDETR, RFDETRBaseConfig]:
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
    elif model_name == "RFDETRXLarge":
        primitive = Bbox
        model_class = RFDETRXLarge
        model_config: config.RFDETRBaseConfig = RFDETRXLargeConfig()
    elif model_name == "RFDETR2XLarge":
        primitive = Bbox
        model_class = RFDETR2XLarge
        model_config: config.RFDETRBaseConfig = RFDETR2XLargeConfig()
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
    return primitive, model, model_config


def to_bbox_primitives(
    predictions,
    image_shape: tuple[int, int],
    bbox_task: TaskInfo
) -> list[Bbox]:
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

# endregion
