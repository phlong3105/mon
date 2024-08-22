#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Value Annotation.

This module implements annotations that take the form of a value (number,
boolean, etc.).
"""

from __future__ import annotations

__all__ = [
    "RegressionAnnotation",
]

import numpy as np
import torch

from mon.core.data.annotation import base


# region Regression

class RegressionAnnotation(base.Annotation):
    """A single regression value.
    
    Args:
        value: The regression value.
        confidence: A confidence in ``[0.0, 1.0]`` for the regression.
            Default: ``1.0``.
    """
    
    def __init__(
        self,
        value     : float,
        confidence: float = 1.0,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.value      = value
        self.confidence = confidence
    
    @property
    def confidence(self) -> float:
        """The confidence of the bounding box."""
        return self._confidence
    
    @confidence.setter
    def confidence(self, confidence: float):
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"`confidence` must be between ``0.0`` and ``1.0``, "
                             f"but got {confidence}.")
        self._confidence = confidence
    
    @property
    def data(self) -> list | None:
        return [self.value]
    
    @staticmethod
    def to_tensor(data: torch.Tensor | np.ndarray, *args, **kwargs) -> torch.Tensor:
        """Converts the input data to a :obj:`torch.Tensor`.
        
        Args:
            data: The input data.
        """
        return torch.Tensor(data)
    
    @staticmethod
    def collate_fn(
        batch: list[torch.Tensor | np.ndarray]
    ) -> torch.Tensor | np.ndarray | None:
        """Collate function used to fused input items together when using
        :obj:`batch_size` > 1. This is used in :obj:`torch.utils.data.DataLoader`
        wrapper.
        
        Args:
            batch: A :obj:`list` of values.
        """
        if all(isinstance(b, torch.Tensor) for b in batch):
            return torch.cat(batch, dim=0)
        elif all(isinstance(b, np.ndarray) for b in batch):
            return np.concatenate(batch, axis=0)
        else:
            return None
        
# endregion
