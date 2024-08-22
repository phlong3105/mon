#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Bounding Box Annotation.

This module implements bounding box annotations in an image.
"""

from __future__ import annotations

__all__ = [
    "BBoxAnnotation",
    "BBoxesAnnotation",
]

from abc import ABC

import numpy as np
import torch

from mon.core.data.annotation import base


# region BBox

class BBoxAnnotation(base.Annotation):
    """A bounding box annotation in an image. Usually, it has a bounding box and
    an instance segmentation mask.
    
    Args:
        class_id: A class ID of the bounding box. ``-1`` means unknown.
        bbox: A bounding box's coordinates of shape ``[4]``.
        confidence: A confidence in ``[0.0, 1.0]`` for the detection.
            Default: ``1.0``.
        
    """
    
    def __init__(
        self,
        class_id  : int,
        bbox      : np.ndarray | list | tuple,
        confidence: float = 1.0,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.class_id   = class_id
        self.bbox       = bbox
        self.confidence = confidence
    
    @property
    def bbox(self) -> np.ndarray:
        """Return the bounding box of shape `[4]`."""
        return self._bbox
    
    @bbox.setter
    def bbox(self, bbox: np.ndarray | list | tuple):
        bbox = np.ndarray(bbox) if isinstance(bbox, list | tuple) else bbox
        if bbox.ndim == 1 and bbox.size == 4:
            self._bbox = bbox
        else:
            raise ValueError(
                f"`bbox` must be a 1D array of size ``4``, but got "
                f"{bbox.ndim} and {bbox.size}."
            )
    
    @property
    def confidence(self) -> float:
        """The confidence of the bounding box."""
        return self._confidence
    
    @confidence.setter
    def confidence(self, confidence: float):
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(
                f"`confidence` must be between ``0.0`` and ``1.0``, "
                f"but got {confidence}."
            )
        self._confidence = confidence
    
    @property
    def data(self) -> list | None:
        """The label's data."""
        return [
            self.bbox[0],
            self.bbox[1],
            self.bbox[2],
            self.bbox[3],
            self.confidence,
            self.class_id,
        ]
    
    @staticmethod
    def to_tensor(data: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Converts the input data to a :obj:`torch.Tensor`.
        
        Args:
            data: The input data.
        """
        return torch.Tensor(data)
    
    @staticmethod
    def collate_fn(batch: list[torch.Tensor | np.ndarray]) -> torch.Tensor | np.ndarray | None:
        """Collate function used to fused input items together when using
        :obj:`batch_size` > 1. This is used in :obj:`torch.utils.data.DataLoader`
        wrapper.
        
        Args:
            batch: A :obj:`list` of images.
        """
        if all(isinstance(b, torch.Tensor) for b in batch):
            return torch.cat(batch, dim=0)
        elif all(isinstance(b, np.ndarray) for b in batch):
            return np.concatenate(batch, axis=0)
        else:
            return None
    

class BBoxesAnnotation(base.Annotation, ABC):
    """A list of all bounding box annotations in an image.
    
    Args:
    
    """
    
    def __init__(self):
        super().__init__()
        self.annotations: list[BBoxAnnotation] = []
    
    @property
    def data(self) -> list | None:
        return [i.data for i in self.annotations]
    
    @property
    def class_ids(self) -> list[int]:
        return [i.class_id for i in self.annotations]
    
    @property
    def bboxes(self) -> list:
        return [i.bbox for i in self.annotations]
    
    @property
    def confidences(self) -> list[float]:
        return [i.confidence for i in self.annotations]
    
# endregion
