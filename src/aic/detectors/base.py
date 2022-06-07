#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Base class for all detector models. It defines an unify template to
guarantee the input and output of all object_detectors are the same.
"""

from __future__ import annotations

import abc
from typing import Optional

import numpy as np
from torch import Tensor

from aic.objects.detection import Detection
from one import ClassLabels
from one import Int3T
from one import select_device

__all__ = [
    "BaseDetector",
]


# MARK: - BaseDetector

class BaseDetector(metaclass=abc.ABCMeta):
    """Base Detector.

    Attributes:
        name (str):
            Name of the detector model.
        model (nn.Module):
            Detector model.
        model_cfg (dict, optional):
            Detector model's config.
        class_labels (ClassLabels, optional):
            List of all labels' dicts.
        allowed_ids (np.ndarray, optional):
            Array of all class_labels' ids. Default: `None`.
        weights (str, optional):
            Path to the pretrained weights. Default: `None`.
        shape (Int3T, optional):
            Input size as [H, W, C]. This is also used to resize the image.
            Default: `None`.
        min_confidence (float, optional):
            Detection confidence threshold. Remove all detections that have a
            confidence lower than this value. If `None`, don't perform
            suppression. Default: `0.5`.
        nms_max_overlap (float, optional):
            Maximum measurement overlap (non-maxima suppression threshold).
            If `None`, don't perform suppression. Default: `0.4`.
        device (str, optional):
            Cuda device, i.e. 0 or 0,1,2,3 or cpu. Default: `None` means CPU.
        resize_original (bool):
            Should resize the predictions back to original image resolution?
            Default: `False`.
    """

    # MARK: Magic Functions

    def __init__(
        self,
        name           : str,
        model_cfg      : Optional[dict],
        class_labels   : ClassLabels,
        weights        : Optional[str]   = None,
        shape          : Optional[Int3T] = None,
        min_confidence : Optional[float] = 0.5,
        nms_max_overlap: Optional[float] = 0.4,
        device         : Optional[str]   = None,
        *args, **kwargs
    ):
        super().__init__()
        self.name            = name
        self.model           = None
        self.model_cfg       = model_cfg
        self.class_labels    = class_labels
        self.allowed_ids     = self.class_labels.ids(key="train_id")
        self.weights         = weights
        self.shape           = shape
        self.min_confidence  = min_confidence
        self.nms_max_overlap = nms_max_overlap
        self.device          = select_device(device)
        self.resize_original = False
        self.half            = self.device.type != "cpu"

        # NOTE: Load model
        self.init_model()

    # MARK: Configure

    @abc.abstractmethod
    def init_model(self):
        """Create and load model from weights."""
        pass

    # MARK: Process

    def detect(self, indexes: np.ndarray, images: np.ndarray) -> list[list[Detection]]:
        """Detect objects in the images.

        Args:
            indexes (np.ndarray):
                Image indexes.
            images (np.ndarray[B, H, W, C]):
                Images.

        Returns:
            detections (list):
                List of `Detection` objects.
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
        detections = self.suppress_wrong_labels(detections=detections)

        return detections

    @abc.abstractmethod
    def forward(self, input: Tensor) -> Tensor:
        """Forward pass.

        Args:
            input (Tensor[B, C, H, W]):
                Input.

        Returns:
            pred (Tensor):
                Predictions.
        """
        pass

    @abc.abstractmethod
    def preprocess(self, images: np.ndarray) -> Tensor:
        """Preprocess the input images to model's input image.

        Args:
            images (np.ndarray[B, H, W, C]):
                Images.

        Returns:
            input (Tensor):
                Models' input.
        """
        pass

    @abc.abstractmethod
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
                Images.
            input (Tensor[B, C, H, W]):
                Input.
            pred (Tensor):
               Prediction.

        Returns:
            detections (list):
                List of `Detection` objects.
        """
        pass

    def suppress_wrong_labels(self, detections: list) -> list[Detection]:
        """Suppress all detections with wrong labels.

        Args:
            detections (list):
                List of `Detection` objects of shape [B, ...], where B is the
                number of batch.

        Returns:
            valid_detections (list):
                List of valid `Detection` objects of shape [B, ...], where B
                is the number of batch.
        """
        valid_detections = []
        for dets in detections:
            valid_dets = [d for d in dets if self.is_correct_label(d.class_label)]
            valid_detections.append(valid_dets)
        return valid_detections

    def suppress_low_confident(self, detections: list) -> list[Detection]:
        """Suppress detections with low confidence scores.

        Args
            detections (list):
                List of valid `Detection` objects of shape [B, ...], where B is
                the number of batch.

        Returns:
            valid_detections (list):
                List of high-confident `Detection` objects of shape [B, ...],
                where B is the number of batch.
        """
        if self.min_confidence is None:
            return detections
        
        valid_detections = []
        for dets in detections:
            valid_dets = [d for d in dets if (d.confidence >= self.min_confidence)]
            valid_detections.append(valid_dets)
        return valid_detections

    def suppress_non_max(self, detections: list) -> list[Detection]:
        """Suppress overlapping detections (high-level). Original code from:
        .. [1] http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/

        Args:
            detections (list):
                List of high-confident `Detection` objects of shape [B, ...],
                where B is the number of batch.

        Returns:
            valid_detections (list):
                List of non-overlapped `Detection` objects of shape [B, ...],
                where B is the number of batch.
        """
        if self.nms_max_overlap is None:
            return detections
        
        valid_detections = []
        for dets in detections:
            # NOTE: Extract measurement bounding boxes and scores
            boxes  = np.array([d.box        for d in dets])
            scores = np.array([d.confidence for d in dets])

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
                            [last], np.where(overlap > self.nms_max_overlap)[0]
                        ))
                    )

            # NOTE: Get exactly the vehicles surviving non-max-suppression
            valid_dets = [dets[i] for i in indices]
            valid_detections.append(valid_dets)

        return valid_detections

    # MARK: Utils

    def is_correct_label(self, label: dict) -> bool:
        """Check if the label is allowed in our application.

        Args:
            label (dict):
                Label dict.

        Returns:
            True or false.
        """
        if label["id"] in self.allowed_ids:
            return True
        return False
