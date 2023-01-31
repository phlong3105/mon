#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""YOLOv5_v4 object_detectors.
"""

from __future__ import annotations

import os
from collections import OrderedDict

import numpy as np
import torch
from torch import Tensor

from aic.builder import DETECTORS
from aic.detectors.base import BaseDetector
from aic.objects.detection import Detection
from aic.utils import pretrained_dir
from onevision import check_image_size
from onevision import is_torch_saved_file
from onevision import is_yaml_file
from onevision import letterbox_resize
from onevision import load_state_dict_from_path
from onevision import scale_box_original
from onevision import to_tensor
from onevision.vision.detection.yolov5_v4.models import yolo
from onevision.vision.detection.yolov5_v4.utils.general import non_max_suppression

__all__ = [
    "YOLOv5_v4",
]


# MARK: - YOLOv5_v4

@DETECTORS.register(name="yolov5_v4")
class YOLOv5_v4(BaseDetector):
    """YOLOv5_v4 object detector."""

    # MARK: Magic Functions

    def __init__(self, name: str = "yolov5_v4", *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.shape = check_image_size(self.shape, 32)  # check img_size

    # MARK: Configure

    def init_model(self):
        """Create and load model from weights."""
        # NOTE: Create model
        if self.model is None:
            assert hasattr(self.model_cfg, "cfg")
            cfg = self.model_cfg.get("cfg")
            nc  = self.model_cfg.get("nc", self.class_labels.num_classes())
            if not is_yaml_file(path=cfg):
                model_dir = os.path.dirname(os.path.abspath(yolo.__file__))
                cfg, _    = os.path.splitext(os.path.basename(cfg))
                cfg       = os.path.join(model_dir, f"{cfg}.yaml")
            self.model = yolo.Model(cfg=cfg, nc=nc)

        # NOTE: Load weights/weights
        if self.weights:
            path = self.weights
            if not is_torch_saved_file(path):
                path, _ = os.path.splitext(os.path.basename(path))
                path    = os.path.join(pretrained_dir, self.name, f"{path}.pt")
            if not is_torch_saved_file(path):
                raise ValueError(f"Not a weights file: {path}.")

            #ckpt       = torch.load(path)
            #state_dict = adjust_state_dict(ckpt["model_state_dict"])
            state_dict = load_state_dict_from_path(path)
            state_dict = adjust_state_dict(state_dict)
            self.model.load_state_dict(state_dict, strict=False)
        
        # NOTE: Eval
        self.model.to(device=self.device)
        self.model.eval()
        
    # MARK: Process

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass.

        Args:
            input (Tensor[B, C, H, W]):
                Input.

        Returns:
            pred (Tensor):
                Predictions.
        """
        # Currently, the only way to avoid memory leak
        with torch.no_grad():
            pred = self.model(input, augment=False)[0]
            pred = non_max_suppression(
                prediction = pred,
                conf_thres = self.min_confidence,
                iou_thres  = self.nms_max_overlap,
                classes    = self.allowed_ids
            )
            return pred

    def preprocess(self, images: np.ndarray) -> Tensor:
        """Preprocess the input images to model's input image.

        Args:
            images (np.ndarray[B, H, W, C]):
                Images.

        Returns:
            input (Tensor[B, C, H, W]):
                Models' input.
        """
        input = images
        if self.shape:
            if input.ndim == 4:
                input = [letterbox_resize(i, size=self.shape, stride=32)[0] for i in input]
                input = np.array(input)
            else:
                input = letterbox_resize(input, size=self.shape, stride=32)[0]
            input = np.ascontiguousarray(input)
            # input = padded_resize(input, self.shape)
            self.resize_original = True
        # input = [F.to_tensor(i) for i in input]
        # input = torch.stack(input)
        input = to_tensor(input, normalize=True)
        input = input.to(self.device)
        return input
    
    def postprocess(
        self,
        indexes: np.ndarray,
        images : np.ndarray,
        input  : Tensor,
        pred   : Tensor,
        *args, **kwargs
    ) -> list:
        """Postprocess the prediction.

        Args:
            indexes (np.ndarray):
                Image indexes.
            images (np.ndarray[B, H, W, C]):
                Images .
            input (Tensor[B, C, H, W]):
                Input.
            pred (Tensor):
                Prediction.

        Returns:
            detections (list):
                List of `Instances` objects.
        """
        # NOTE: Resize
        if self.resize_original:
            for det in pred:
                det[:, :4] = scale_box_original(
                    box      = det[:, :4],
                    cur_size = input.shape[2:],
                    new_size = images.shape[1:3]
                ).round()

        # NOTE: Create Detection objects
        detections = []
        for idx, det in enumerate(pred):
            inst = []
            for *xyxy, conf, cls in det:
                box_xyxy    = np.array([xyxy[0].item(), xyxy[1].item(),
                                     xyxy[2].item(), xyxy[3].item()], np.int32)
                confident   = float(conf)
                class_id    = int(cls)
                class_label = self.class_labels.get_class_label(key="train_id", value=class_id)
                inst.append(
                    Detection(
                        frame_index = indexes[0] + idx,
                        box         = box_xyxy,
                        confidence  = confident,
                        class_label = class_label
                    )
                )
            detections.append(inst)
        return detections


# MARK: - Utils

def adjust_state_dict(state_dict: OrderedDict) -> OrderedDict:
    od = OrderedDict()
    for key, value in state_dict.items():
        new_key     = key.replace("module.", "")
        od[new_key] = value
    return od
