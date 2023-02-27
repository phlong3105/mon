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
        to_instance: If True, wrap the predictions to a list of
            :class:`supr.data.instance.Instance` object. Else, return raw
            predictions. Defaults to True.
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
        to_instance   : bool  = True,
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
        self.to_instance    = to_instance
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
    ) -> list[list[mon.Instance]] | list[np.ndarray]:
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
    ) -> list[list[mon.Instance]]:
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
    
# endregion
