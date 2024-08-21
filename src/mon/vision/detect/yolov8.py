#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements YOLOv8 detectors."""

from __future__ import annotations

__all__ = [
    "YOLOv8Detector",
]

import sys

import numpy as np
import torch

from mon import core
from mon.globals import DETECTORS
from mon.vision import track
from mon.vision.detect import base

console       = core.console
error_console = core.error_console
current_file  = core.Path(__file__).absolute()
current_dir   = current_file.parents[0]

try:
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.nn import tasks
    from ultralytics.yolo.data import augment
    from ultralytics.yolo.utils import checks, ops
    ultralytics_available = True
except ImportError:
    ultralytics_available = False
    error_console.log("The package 'ultralytics' has not been installed.")
    # sys.exit(1)  # Exit and raise error
    sys.exit(0)  # Exit without error


# region YOLOv8

@DETECTORS.register(name="yolov8")
@DETECTORS.register(name="yolov8l")
@DETECTORS.register(name="yolov8m")
@DETECTORS.register(name="yolov8n")
@DETECTORS.register(name="yolov8s")
@DETECTORS.register(name="yolov8x")
class YOLOv8Detector(base.Detector):
    """YOLOv8 detector.
    
    See Also: :class:`mon.vision.detect.base.Detector`.
    """
    
    def __init__(
        self,
        config : dict | str | core.Path | None = current_dir / "config/yolov8.yaml",
        *args, **kwargs
    ):
        super().__init__(config=config, *args, **kwargs)
        self.model = YOLO(model=str(self.weights), task="detect", verbose=False)
        
    def __call__(
        self,
        indexes: np.ndarray | list[int],
        images : str | core.Path | list[str] | list[core.Path] | np.ndarray | torch.Tensor,
        **kwargs,
    ) -> np.ndarray:
        """Detect objects in the images.

        Args:
            indexes: Image indexes of shape :math:`[B]`.
            images: The source of the image(s) to make predictions on. Accepts
                various types including file paths, URLs, PIL images, numpy
                arrays, and torch tensors.
            **kwargs: Additional keyword arguments for configuring the prediction
                process.
        
        Returns:
            A 2D :class:`numpy.ndarray` or :class:`torch.Tensor` of detections.
            The most common format is :math:`[B, N, 6]` where :math:`B` is the
            batch size, :math:`N` is the number of detections, and :math:`[6]`
            usually contains :math:`[x1, y1, x2, y2, conf, class_id]`. Notice
            that :math:`[x1, y1, x2, y2]` should be scaled back to the original
            image size.
            
        Examples:
            >>> model   = YOLO('yolov8n.pt')
            >>> results = model.predict(source='path/to/image.jpg', conf=0.25)
            >>> for r in results:
            >>>     print(r.boxes.data)  # print detection bounding boxes
        """
        # Prepare input
        indexes = list(indexes)
        if isinstance(images, (str, core.Path)):
            images = [images]
        elif isinstance(images, np.ndarray) and images.ndim == 4:
            images = list(images)
        
        # Overwrite model's configs
        config  = self.config | kwargs
        
        # Make predictions
        outputs = self.model.predict(source=images, **config)
        
        # Obtain results
        results = []
        for r in outputs:
            results.append(r.boxes.data)
        return results
        
        
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
        ratio  = max(self.image_size) / max(core.get_image_size(image=input))
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
            image= input,
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
        h0, w0 = core.get_image_size(image=images)
        h1, w1 = core.get_image_size(image=input)
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
