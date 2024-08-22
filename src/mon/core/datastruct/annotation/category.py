#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements annotations that take the form of a category or class."""

from __future__ import annotations

__all__ = [
    "ClassificationAnnotation",
    "class_id_to_logits",
    "logits_to_class_id",
]

import numpy as np
import torch

from mon import core
from mon.data.datastruct.annotation import base

console = core.console


# region Utils

def logits_to_class_id(logits: np.ndarray) -> np.ndarray:
    """Convert logits to class IDs.

    Args:
        logits: A :obj:`numpy.ndarray` of logits where each row corresponds
            to a set of logits for a sample.

    Returns:
        A :obj:`numpy.ndarray` of class IDs corresponding to the highest logit
        values for each sample.
    """
    # Use np.argmax to get the index of the maximum value along the last axis (axis=1)
    class_id = np.argmax(logits, axis=-1)
    return class_id


def class_id_to_logits(
    class_id   : int,
    num_classes: int,
    high_value : float = 1.0,
    low_value  : float = 0.0
) -> np.ndarray:
    """Convert a class ID to logits.

    Args:
        class_id: The ID of the class for which to generate the logit.
        num_classes: The total number of classes in the classification task.
        high_value: The logit value for the target class.
        low_value: The logit value for non-target classes.

    Returns:
        A :obj:`numpy.ndarray` represents the logits for the given class ID.
    """
    # Initialize logits with low_value
    logits = np.full(num_classes, low_value)
    # Set the logit for the target class ID to high_value
    logits[class_id] = high_value
    return logits

# endregion


# region Classification

class ClassificationAnnotation(base.Annotation):
    """A classification annotation for an image.
    
    Args:
        class_id: A class ID of the classification data. ``-1`` means unknown.
        num_classes: The total number of classes in the classification task.
        confidence: A confidence in ``[0.0, 1.0]`` for the classification.
            Default: ``1.0``.
        
    """
    
    def __init__(
        self,
        class_id   : int,
        num_classes: int,
        confidence : float = 1.0,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.class_id    = class_id
        self.num_classes = num_classes
        self.confidence  = confidence
        self.logits	     = class_id_to_logits(class_id=class_id, num_classes=num_classes)
    
    @property
    def confidence(self) -> float:
        """The confidence of the bounding box."""
        return self._confidence
    
    @confidence.setter
    def confidence(self, confidence: float):
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"`confidence` must be between ``0.0`` and ``1.0``, but got {confidence}.")
        self._confidence = confidence
    
    @property
    def data(self) -> list | None:
        return [self.class_id]
    
    @staticmethod
    def to_tensor(data: torch.Tensor | np.ndarray, *args, **kwargs) -> torch.Tensor:
        """Converts the input data to a :obj:`torch.Tensor`.
        
        Args:
            data: The input data.
        """
        return torch.Tensor(data)
    
    @staticmethod
    def collate_fn(batch: list[torch.Tensor | np.ndarray]) -> torch.Tensor | np.ndarray | None:
        """Collate function used to fused input items together when using
        :obj:`batch_size` > 1. This is used in :obj:`torch.utils.data.DataLoader` wrapper.
        
        Args:
            batch: A :obj:`list` of class ids.
        """
        if all(isinstance(b, torch.Tensor) for b in batch):
            return torch.cat(batch, dim=0)
        elif all(isinstance(b, np.ndarray) for b in batch):
            return np.concatenate(batch, axis=0)
        else:
            return None
    
# endregion
