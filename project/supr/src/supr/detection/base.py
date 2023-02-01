#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the base class for all detectors."""

from __future__ import annotations

__all__ = [
    "Detector",
]

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from torch import Tensor

import mon
from supr.data import instance as ins

if TYPE_CHECKING:
    from supr.typing import (
        ClassLabelsType, ConfigType, DictType, Int3T, Ints, Strs, WeightsType
    )


# region Detector

class Detector(ABC):
    """The base class for all detectors.

    Args:
        cfg: A detector model's config.
        classlabels: A list of all labels' dicts.
        weights: A path to the pretrained weights. Defaults to None.
        shape: The desired model's input shape preferably in a channel-last
            format. Defaults to (3, 256, 256).
        conf_thres: An object confidence threshold. Defaults to 0.5.
        iou_thres: An IOU threshold for NMS. Defaults to 0.4.
        devices: A list of devices to use. Defaults to 'cpu'.
    """
    
    def __init__(
        self,
        cfg        : ConfigType,
        classlabels: ClassLabelsType,
        weights    : WeightsType | None = None,
        shape      : Int3T              = (3, 256, 256),
        conf_thres : float              = 0.5,
        iou_thres  : float              = 0.4,
        devices    : Ints | Strs        = "cpu",
    ):
        super().__init__()
        self.model           = None
        self.cfg             = cfg
        self.classlabels     = mon.ClassLabels.from_value(value=classlabels)
        self.allowed_ids     = self.classlabels.ids(key="train_id")
        self.weights         = weights
        self.shape           = shape
        self.conf_thres      = conf_thres
        self.iou_thres       = iou_thres
        self.devices         = mon.select_device(device=devices)
        self.half            = self.devices.type != "cpu"
        self.resize_original = False
        # Load model
        self.init_model()
    
    @abstractmethod
    def init_model(self):
        """Create model."""
        pass

    def detect(self, indexes: np.ndarray, images: np.ndarray) -> list[list[ins.Instance]]:
        """Detect objects in the images.

        Args:
            indexes: Image indexes.
            images: Images.

        Returns:
            A list of :class:`data.Instance` objects.
        """
        # NOTE: Safety check
        if self.model is None:
            raise NotImplementedError(f"Model has not been defined yet!")
        # NOTE: Preprocess
        input      = self.preprocess(images=images)
        # NOTE: Forward
        pred       = self.forward(input)
        # NOTE: Postprocess
        detections = self.postprocess(indexes=indexes, images=images, input=input, pred=pred)
        # NOTE: Suppression
        detections = self.suppress_wrong_labels(instances=detections)

        return detections

    @abstractmethod
    def forward(self, input: Tensor) -> Tensor:
        """Forward pass.

        Args:
            input: Input tensor of shape [B, C, H, W].

        Returns:
            Predictions.
        """
        pass

    @abstractmethod
    def preprocess(self, images: np.ndarray) -> Tensor:
        """Preprocess the input images to model's input image.

        Args:
            images: Images.

        Returns:
            Input tensor of shape [B, H, W, C].
        """
        pass

    @abstractmethod
    def postprocess(
        self,
        indexes: np.ndarray,
        images : np.ndarray,
        input  : Tensor,
        pred   : Tensor,
        *args, **kwargs
    ) -> list:
        """Postprocess predictions.

        Args:
            indexes: Image indexes.
            images: Images.
            input: Input tensor.
            pred: Predictions.

        Returns:
            A 2-D list of :class:`data.Instance` objects.
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
            valid_ins = [d for d in ins if self.is_correct_label(d.class_label)]
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
        if self.conf_thres is None:
            return instances
        
        valid_instances = []
        for ins in instances:
            valid_ins = [d for d in ins if (d.confidence >= self.conf_thres)]
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
        if self.iou_thres is None:
            return instances
        
        valid_instances = []
        for ins in instances:
            # NOTE: Extract measurement bounding boxes and scores
            boxes  = np.array([d.box        for d in ins])
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
                            [last], np.where(overlap > self.iou_thres)[0]
                        ))
                    )

            # Get exactly the vehicles surviving non-max-suppression
            valid_ins = [ins[i] for i in indices]
            valid_instances.append(valid_ins)

        return valid_instances

    def is_correct_label(self, label: DictType) -> bool:
        """Check if the label is allowed."""
        if label["id"] in self.allowed_ids:
            return True
        return False

# endregion
