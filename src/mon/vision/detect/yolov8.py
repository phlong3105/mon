#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements YOLOv8 detectors."""

from __future__ import annotations

__all__ = [
    "YOLOv8",
]

import numpy as np
import torch
from ultralytics.nn import tasks
from ultralytics.yolo.data import augment
from ultralytics.yolo.utils import checks, ops

from mon import core
from mon.globals import DETECTORS
from mon.vision import track
from mon.vision.detect import base

console = core.console


# region YOLOv8

@DETECTORS.register(name="yolov8")
class YOLOv8(base.Detector1):
    """YOLOv8 detector.
    
    See Also: :class:`mon.vision.detect.base.Detector`.
    """
    
    def init_model(self):
        """Create model."""
        self.model = tasks.attempt_load_weights(
            weights = str(self.weights),
            device  = self.device
        )
        self.image_size = checks.check_imgsz(
            imgsz   = self.image_size,
            stride  = self.model.stride,
            min_dim = 2
        )
    
    def preprocess(self, images: np.ndarray) -> torch.Tensor:
        """Preprocessing step.

        Args:
            images: Images of shape :math:`[B, C, H, W]`.

        Returns:
            Input tensor of shape :math:`[B, C, H, W]`.
        """
        input  = images.copy()
        ratio  = max(self.image_size) / max(
            core.get_image_size(input=input))
        stride = self.model.stride
        stride = int(stride.max() if isinstance(stride, torch.Tensor) else stride)
        
        if ratio != 1:
            letterbox = augment.LetterBox(
                new_shape = self.image_size,
                auto      = True,
                stride    = stride
            )
            if input.ndim == 4:
                input = [letterbox(image=i) for i in input]
            elif input.ndim == 3:
                input = letterbox(image=input)
            else:
                raise ValueError
            input = np.ascontiguousarray(input)

        input = core.to_image_tensor(
            input= input,
            keepdim   = False,
            normalize = True,
            device    = self.device
        )
        return input
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            input: Input tensor of shape :math:`[B, C, H, W]`.
        
        Returns:
            Predictions.
        """
        return self.model.forward(input)
    
    def postprocess(
        self,
        indexes: np.ndarray,
        images : np.ndarray,
        input  : torch.Tensor,
        pred   : torch.Tensor,
        *args, **kwargs
    ) -> list[np.ndarray] | list[list[track.Instance]]:
        """Postprocessing step.

        Args:
            indexes: A :class:`list` of image indexes.
            images: Images of shape :math:`[B, C, H, W]`.
            input: Input tensor of shape :math:`[B, C, H, W]`.
            pred: Prediction tensor of shape :math:`[B, C, H, W]`.

        Returns:
            A 2D :class:`list` of :class:`data.Instance` objects. The outer
            :class:`list` has ``B`` items.
        """
        pred = ops.non_max_suppression(
            prediction = pred,
            conf_thres = self.conf_threshold,
            iou_thres  = self.iou_threshold,
            agnostic   = False,
            max_det    = self.max_detections,
            classes    = self.allowed_ids
        )
        h0, w0 = core.get_image_size(input=images)
        h1, w1 = core.get_image_size(input=input)
        for i, p in enumerate(pred):
            p[:, :4]  = ops.scale_boxes((h1, w1), p[:, :4], (h0, w0)).round()
            p         = p.detach().cpu().numpy()
            pred[i]   = p
            
        if self.to_instance:
            batch_instances = []
            for i, p in enumerate(pred):
                instances = []
                for *xyxy, conf, cls in p:
                    classlabel = self.classlabels.get_class(key="id", value=cls)
                    instances.append(
                        track.Instance(
                            bbox        = xyxy,
                            confidence  = conf,
                            classlabel  = classlabel,
                            frame_index = indexes[0] + i,
                        )
                    )
                batch_instances.append(instances)
            return batch_instances
        else:
            return pred
    
# endregion
