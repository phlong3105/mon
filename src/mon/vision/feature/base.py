#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the base class for embedders."""

from __future__ import annotations

__all__ = [
    "DeepEmbedder", "Embedder",
]

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch

from mon.foundation import pathlib
from mon.vision import image as mimage
from mon.vision.ml import device as mdevice


# region Embedder

class Embedder(ABC):
    """The base class for all embedders."""

    @abstractmethod
    def embed(self, *args, **kwargs) -> list[np.ndarray]:
        """Extract features in the images.

        Args:
            indexes: A list of image indexes.
            images: Images of shape NHWC.

        Returns:
           A 2-D list of feature vectors.
        """
        pass
        
    
class DeepEmbedder(Embedder, ABC):
    """The base class for all deep learning-based embedders. It loads a
    classification model pretrained on Imagenet, with classification layer
    removed, exposing the bottleneck layer, and outputting a feature.
    
    Args:
        config: A detector model's config.
        weight: A path to a pretrained weight file.
        image_size: The desired model's input size in HW format. Defaults to
            640.
        device: Cuda device, i.e. 0 or 0,1,2,3 or cpu. Defaults to 'cpu'.
        to_numpy: If True, convert the embedded feature vectors to
            :class:`np.ndarray`. Defaults to False.
    
    See Also: :class:`Embedder`.
    """
    
    def __init__(
        self,
        config    : dict | pathlib.Path | None,
        weight    : Any,
        image_size: int | list[int] = 224,
        device    : int | str | list[int | str] = "cpu",
        to_numpy  : bool = False,
        *args, **kwargs
    ):
        super().__init__()
        self.config     = config
        self.weight     = weight
        self.image_size = mimage.get_hw(size=image_size)
        self.device     = mdevice.select_device(device=device)
        self.to_numpy   = to_numpy
        # Load model
        self.model = None
        self.init_model()
    
    @property
    def weight(self):
        return self._weight
    
    @weight.setter
    def weight(self, weight: Any):
        if isinstance(weight, pathlib.Path | str):
            weight = pathlib.Path(weight)
            if not weight.is_torch_file():
                raise ValueError(
                    f"weight must be a valid path to a torch saved file, but "
                    f"got {weight}."
                )
        elif isinstance(weight, list | tuple):
            weight = [pathlib.Path(w) for w in weight]
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
    
    def embed(self, indexes: np.ndarray, images : np.ndarray) -> list[np.ndarray]:
        """Extract features in the images.

        Args:
            indexes: A list of image indexes.
            images: Images of shape NHWC.

        Returns:
           A 2-D list of feature vectors.
        """
        if self.model is None:
            raise ValueError(f"model has not been defined yet!")
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
    ) -> list[np.ndarray]:
        """Postprocessing step.

        Args:
            indexes: A list of image indexes.
            images: Images of shape NHWC.
            input: Input tensor of shape NCHW.
            pred: Prediction tensor of shape NCHW.

        Returns:
            A 2-D list of feature vectors.
        """
        pass
    
# endregion
