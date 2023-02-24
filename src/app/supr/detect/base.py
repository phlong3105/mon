#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the base class for all detectors."""

from __future__ import annotations

__all__ = [
    "Detector",
]

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch

import mon
from supr.data import instance as ins


# region Detector

class Detector(ABC):
    """The base class for all detectors.

    Args:
        config: A detector model's config.
        weight: A path to a pretrained weight file.
        classlabels: A list of all the class-labels defined in a dataset.
        image_size: The desired model's input size in HW format. Defaults to
            640.
        conf_threshold: An object confidence threshold. Defaults to 0.5.
        iou_threshold: An IOU threshold for NMS. Defaults to 0.4.
        max_detections: Maximum number of detections/image. Defaults to 300.
        device: Cuda device, i.e. 0 or 0,1,2,3 or cpu. Defaults to 'cpu'.
    """
    
    def __init__(
        self,
        config        : dict | mon.Path | None,
        weight        : Any,
        image_size    : int | list[int] = 640,
        classlabels   : Any   = None,
        conf_threshold: float = 0.5,
        iou_threshold : float = 0.4,
        max_detections: int   = 300,
        device        : int | str | list[int | str] = "cpu",
    ):
        super().__init__()
        self.config         = config
        self.weight         = weight
        self.classlabels    = mon.ClassLabels.from_value(value=classlabels)
        self.allowed_ids    = self.classlabels.ids(key="id", exclude_negative_key=True)
        self.image_size     = mon.get_hw(size=image_size)
        self.conf_threshold = conf_threshold
        self.iou_threshold  = iou_threshold
        self.max_detections = max_detections
        self.device         = mon.select_device(device=device)
        # Load model
        self.model = None
        self.init_model()
    
    @property
    def weight(self):
        return self._weight
    
    @weight.setter
    def weight(self, weight: Any):
        if isinstance(weight, mon.Path | str):
            weight = mon.Path(weight)
            if not weight.is_torch_file():
                raise ValueError(
                    f"weight must be a valid path to a torch saved file, but "
                    f"got {weight}."
                )
        elif isinstance(weight, list | tuple):
            weight = [mon.Path(w) for w in weight]
            if not all(w.is_torch_file for w in weight):
                raise ValueError(
                    f"weight must be a valid path to a torch saved file, but "
                    f"got {weight}."
                )
        self._weight = weight
    
    @abstractmethod
    def init_model(self):
        """Create model."""
        pass
    
    def detect(
        self,
        indexes: np.ndarray,
        images : np.ndarray
    ) -> list[list[ins.Instance]]:
        """Detect objects in the images.

        Args:
            indexes: A list of image indexes.
            images: Images of shape NHWC.

        Returns:
            A 2-D list of :class:`supr.data.Instance` objects. The outer list
            has N items.
        """
        if self.model is None:
            raise ValueError(f"Model has not been defined yet!")
        input     = self.preprocess(images=images)
        pred      = self.forward(input)
        instances = self.postprocess(
            indexes = indexes,
            images  = images,
            input   = input,
            pred    = pred
        )
        return instances

    @abstractmethod
    def preprocess(self, images: np.ndarray) -> torch.Tensor:
        """Preprocessing step.

        Args:
            images: Images of shape NHWC.

        Returns:
            Input tensor of shape NCHW.
        """
        pass

    @abstractmethod
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            input: Input tensor of shape NCHW.

        Returns:
            Predictions.
        """
        pass

    @abstractmethod
    def postprocess(
        self,
        indexes: np.ndarray,
        images : np.ndarray,
        input  : torch.Tensor,
        pred   : torch.Tensor,
        *args, **kwargs
    ) -> list[list[ins.Instance]]:
        """Postprocessing step.

        Args:
            indexes: A list of image indexes.
            images: Images of shape NHWC.
            input: Input tensor of shape NCHW.
            pred: Prediction tensor of shape NCHW.

        Returns:
            A 2-D list of :class:`data.Instance` objects. The outer list has N
            items.
        """
        pass

    def suppress_wrong_labels(
        self,
        instances: list[list[ins.Instance]],
    ) -> list[list[ins.Instance]]:
        """Suppress all detections with wrong labels.

        Args:
            instances: A 2-D list of :class:`data.Instance` objects.

        Returns:
            A 2-D list of valid :class:`data.Instance` objects.
        """
        valid_instances = []
        for ins in instances:
            valid_ins = [d for d in ins if self.is_correct_label(d.classlabel)]
            valid_instances.append(valid_ins)
        return valid_instances
    
    def suppress_low_confident(
        self,
        instances: list[list[ins.Instance]],
    ) -> list[list[ins.Instance]]:
        """Suppress detections with low confidence scores.

        Args
            instances: A 2-D list of :class:`data.Instance` objects.
        
        Returns:
             A 2-D list of :class:`data.Instance` objects.
        """
        if self.conf_threshold is None:
            return instances
        
        valid_instances = []
        for ins in instances:
            valid_ins = [d for d in ins if (d.confidence >= self.conf_threshold)]
            valid_instances.append(valid_ins)
        return valid_instances

    def perform_nms(
        self,
        instances: list[list[ins.Instance]],
    ) -> list[list[ins.Instance]]:
        """Suppress overlapping detections (high-level). Original code from:
        .. [1] http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/

        Args:
            instances: A 2-D list of :class:`data.Instance` objects.

        Returns:
            A 2-D list of non-overlapped :class:`data.Instance` objects.
        """
        if self.iou_threshold is None:
            return instances
        
        valid_instances = []
        for ins in instances:
            # NOTE: Extract measurement bounding boxes and scores
            boxes  = np.array([d.bbox       for d in ins])
            scores = np.array([d.confidence for d in ins])

            # NOTE: Extract road_objects indices that survive
            # non-max-suppression
            indices = []
            if len(boxes) > 0:
                boxes = boxes.astype(np.float)

                # Top-left to Bottom-right
                x1 = boxes[:, 0]
                y1 = boxes[:, 1]
                x2 = boxes[:, 2] + boxes[:, 0]
                y2 = boxes[:, 3] + boxes[:, 1]

                # Area
                area = (boxes[:, 2] + 1) * (boxes[:, 3] + 1)
                if scores is not None:
                    idxs = np.argsort(scores)
                else:
                    idxs = np.argsort(y2)

                # Suppression via iterating boxes
                while len(idxs) > 0:
                    last = len(idxs) - 1
                    i    = idxs[last]
                    indices.append(i)

                    xx1     = np.maximum(x1[i], x1[idxs[:last]])
                    yy1     = np.maximum(y1[i], y1[idxs[:last]])
                    xx2     = np.minimum(x2[i], x2[idxs[:last]])
                    yy2     = np.minimum(y2[i], y2[idxs[:last]])
                    w       = np.maximum(0, xx2 - xx1 + 1)
                    h       = np.maximum(0, yy2 - yy1 + 1)
                    overlap = (w * h) / area[idxs[:last]]
                    idxs    = np.delete(
                        idxs, np.concatenate((
                            [last], np.where(overlap > self.iou_threshold)[0]
                        ))
                    )

            # Get exactly the vehicles surviving non-max-suppression
            valid_ins = [ins[i] for i in indices]
            valid_instances.append(valid_ins)

        return valid_instances

    def is_correct_label(self, label: dict) -> bool:
        """Check if the label is allowed."""
        if label["id"] in self.allowed_ids:
            return True
        return False

# endregion
