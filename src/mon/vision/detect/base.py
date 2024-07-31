#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the base class for all detectors."""

from __future__ import annotations

__all__ = [
    "Detector",
    "Detector1",
]

from abc import ABC, abstractmethod
from typing import Any, Sequence

import numpy as np
import torch

from mon import core, nn
from mon.vision import track

console = core.console


# region Detector

class Detector(ABC):
    """The base class for all detectors. It loads create a detection model,
    load weights, and striping off unnecessary training components.

    Args:
        config: A detector model's config.
        weights: A path to a pretrained weights file.
        image_size: The desired model's input size in :math:`[H, W]` format.
            Default: ``640``.
        conf_threshold: An object confidence threshold. Default: ``0.5``.
        iou_threshold: An IOU threshold for NMS. Default: ``0.3``.
        max_detections: Maximum number of detections/image. Default: ``300``.
        device: Running device, i.e., ``'0'`` or ``'0,1,2,3'`` or ``'cpu'``.
            Default: ``'cpu'``.
    """
    
    def __init__(
        self,
        config        : dict | str | core.Path | None,
        weights       : Any,
        image_size    : int | Sequence[int] = 640,
        conf_threshold: float               = 0.5,
        iou_threshold : float               = 0.3,
        max_detections: int                 = 300,
        device        : int | str           = "cpu",
    ):
        super().__init__()
        self.config         = config
        self.weights        = weights
        self.image_size     = image_size
        self.conf_threshold = conf_threshold
        self.iou_threshold  = iou_threshold
        self.max_detections = max_detections
        self.device         = device
        self.model          = None
    
    # region Properties
    
    @property
    def weights(self) -> core.Path:
        return self._weights
    
    @weights.setter
    def weights(self, weights: Any):
        if isinstance(weights, core.Path | str):
            weights = core.Path(weights)
            if not weights.is_torch_file():
                raise ValueError(
                    f":param:`weights` must be a valid path to a torch saved "
                    f"file, but got {weights}."
                )
        elif isinstance(weights, dict):
            weights = [core.Path(w) for w in weights]
            if not all(w.is_torch_file for w in weights):
                raise ValueError(
                    f":param:`weights` must be a valid path to a torch saved file, but "
                    f"got {weights}."
                )
        else:
            raise ValueError()
        self._weights = weights
    
    @property
    def image_size(self) -> tuple[int, int]:
        return self._image_size
    
    @image_size.setter
    def image_size(self, image_size: int | Sequence[int]):
        self._image_size = core.parse_hw(size=image_size)
    
    @property
    def conf_threshold(self) -> float:
        return self._conf_threshold
    
    @conf_threshold.setter
    def conf_threshold(self, conf_threshold: float):
        if not 0.0 <= conf_threshold <= 1.0:
            raise ValueError(f":param:`conf_threshold` must be between ``0.0`` and ``1.0``, but got {conf_threshold}.")
        self._conf_threshold = conf_threshold
    
    @property
    def iou_threshold(self) -> float:
        return self._iou_threshold
    
    @iou_threshold.setter
    def iou_threshold(self, iou_threshold: float):
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError(f":param:`iou_threshold` must be between ``0.0`` and ``1.0``, but got {iou_threshold}.")
        self._iou_threshold = iou_threshold
    
    @property
    def device(self) -> torch.device:
        return self._device
    
    @device.setter
    def device(self, device: int | str):
        self._device = core.set_device(device)
    
    # endregion
    
    @abstractmethod
    def forward(self, indexes: np.ndarray, images: np.ndarray) -> np.ndarray:
        """Detect objects in the images.

        Args:
            indexes: A :class:`numpy.ndarray` of image indexes of shape :math:`[B]`.
            images: Images of shape :math:`[B, H, W, C]`.

        Returns:
            A 2D :class:`numpy.ndarray` of shape :math:`[B, ..., ..., ...]`.
        """
        pass


class Detector1(ABC):
    """The base class for all detectors. It loads a detection model, striping
    off unnecessary training components.

    Args:
        config: A detector model's config.
        weights: A path to a pretrained weights file.
        classlabels: A :class:`list` of all the class-labels defined in a
            dataset.
        image_size: The desired model's input size in :math:`[H, W]` format.
            Default: ``640``.
        conf_threshold: An object confidence threshold. Default: ``0.5``.
        iou_threshold: An IOU threshold for NMS. Default: ``0.4``.
        max_detections: Maximum number of detections/image. Default: ``300``.
        device: Cuda device, i.e. ``'0'`` or ``'0,1,2,3'`` or ``'cpu'``.
            Default: ``'cpu'``.
        to_instance: If ``True``, wrap the predictions to a :class:`list` of
            :class:`supr.data.instance.Instance` object. Else, return raw
            predictions. Default: ``True``.
    """
    
    def __init__(
        self,
        config        : dict | core.Path | None,
        weights       : Any,
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
        self.weights        = weights
        self.classlabels    = nn.ClassLabels.from_value(value=classlabels)
        self.allowed_ids    = self.classlabels.ids(key="id", exclude_negative_key=True)
        self.image_size     = core.parse_hw(size=image_size)
        self.conf_threshold = conf_threshold
        self.iou_threshold  = iou_threshold
        self.max_detections = max_detections
        self.device         = torch.device(("cpu" if not torch.cuda.is_available() else device))
        self.to_instance    = to_instance
        # Load model
        self.model = None
        self.init_model()
    
    @property
    def weights(self):
        return self._weights
    
    @weights.setter
    def weights(self, weights: Any):
        if isinstance(weights, core.Path | str):
            weights = core.Path(weights)
            if not weights.is_torch_file():
                raise ValueError(
                    f"weights must be a valid path to a torch saved file, but "
                    f"got {weights}."
                )
        elif isinstance(weights, list | tuple):
            weights = [core.Path(w) for w in weights]
            if not all(w.is_torch_file for w in weights):
                raise ValueError(
                    f"weights must be a valid path to a torch saved file, but "
                    f"got {weights}."
                )
        self._weights = weights
    
    @abstractmethod
    def init_model(self):
        """Create model."""
        pass
    
    def detect(
        self,
        indexes: np.ndarray,
        images : np.ndarray
    ) -> list[np.ndarray] | list[list[track.Instance]]:
        """Detect objects in the images.

        Args:
            indexes: A :class:`list` of image indexes.
            images: Images of shape :math:`[B, H, W, C]`.

        Returns:
            A 2D :class:`list` of :class:`supr.data.Instance` objects. The
            outer :class:`list` has ``B`` items.
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
            images: Images of shape :math:`[B, H, W, C]`.

        Returns:
            Input tensor of shape :math:`[B, C, H, W]`.
        """
        pass
    
    @abstractmethod
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            input: Input tensor of shape :math:`[B, C, H, W]`.

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
    ) -> list[np.ndarray] | list[list[track.Instance]]:
        """Postprocessing step.

        Args:
            indexes: A :class:`list` of image indexes.
            images: Images of shape :math:`[B, H, W, C]`.
            input: Input tensor of shape :math:`[B, H, W, C]`.
            pred: Prediction tensor of shape :math:`[B, H, W, C]`.

        Returns:
            A 2D :class:`list` of :class:`data.Instance` objects. The outer
            :class:`list` has ``B`` items.
        """
        pass
    
# endregion
