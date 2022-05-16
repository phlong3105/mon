#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from abc import ABCMeta
from abc import abstractmethod
from typing import Optional

from one.core import Indexes
from one.core import TensorOrArray
from one.core import TensorsOrArrays
from one.nn.model import BaseModel

__all__ = [
    "LaneDetector",
]


# MARK: - Module

class LaneDetector(BaseModel, metaclass=ABCMeta):
    """Base class for all lane detector models."""
    
    # MARK: Magic Functions
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    # MARK: Forward Pass
    
    def forward(
        self, x: TensorOrArray, augment: bool = False, *args, **kwargs
    ) -> TensorOrArray:
        """Forward pass. This is the primary `forward` function of the model.
        It supports augmented inference.
        
        In this function, we perform test-time augmentation and pass the
        transformed input to `forward_once()`.

        Args:
            x (TensorOrArray[B, C, H, W]):
                Input.
            augment (bool):
                Augmented inference. Default: `False`.
                
        Returns:
            yhat (Tensor):
                Predictions.
        """
        if augment:
            # NOTE: For now just forward the input. Later, we will implement
            # the test-time augmentation for image classification
            return self.forward_once(x=x, *args, **kwargs)
        else:
            return self.forward_once(x=x, *args, **kwargs)
    
    @abstractmethod
    def forward_once(self, x: TensorOrArray, *args, **kwargs) -> TensorOrArray:
        """Forward pass once. Implement the logic for a single forward pass.

        Args:
            x (TensorOrArray[B, C, H, W]):
                Input.

        Returns:
            yhat (Tensor):
                Predictions.
        """
        pass
    
    def forward_features(
        self, x: TensorOrArray, out_indexes: Optional[Indexes] = None
    ) -> TensorsOrArrays:
        """Forward pass for features extraction.

        Args:
            x (TensorOrArray[B, C, H, W]):
                Input.
            out_indexes (Indexes, optional):
                List of layers' indexes to extract features. This is called
                in `forward_features()` and is useful when the model
                is used as a component in another model.
                - If is a `tuple` or `list`, return an array of features.
                - If is a `int`, return only the feature from that layer's
                index.
                - If is `-1`, return the last layer's output.
                Default: `None`.
        """
        pass
