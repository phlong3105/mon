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
        config: A detector model's config. It must include all the necessary
            information to create a model. Accepts various types including file
            paths and dictionaries.
        weights: A path to a pretrained weights file.
    """
    
    def __init__(
        self,
        config : dict | str | core.Path | None,
        weights: Any,
        *args, **kwargs
    ):
        super().__init__()
        self.config   = config
        self.config  |= kwargs
        self.weights  = weights
        self.model    = None
    
    # region Properties
    
    @property
    def config(self) -> dict:
        return self._config
    
    @config.setter
    def config(self, config: dict | str | core.Path | None):
        if isinstance(config, dict):
            self._config = config
        elif isinstance(config, core.Path | str):
            config = core.Path(config)
            if not config.is_yaml():
                raise ValueError(f":param:`config` must be a valid path to a YAML file, but got {config}.")
            self._config = core.read_from_file(config)
        else:
            self._config = {}
    
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
        
    # endregion
    
    @abstractmethod
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
