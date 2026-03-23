#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Segmentor Wrappers.

This module provides wrappers for segmentation models to extract masks.
"""

from __future__ import annotations

__all__ = [
    "Segmentor",
    "SAMBoxSegmentor",
]

from abc import ABC, abstractmethod
from typing import Any, override

import cv2
import numpy as np
from numpy import ndarray
from torch import nn

from mon.core import (
    BBoxes,
    DeviceLike,
    Int3,
    MODELS,
    Path,
    Size,
    sys_ctx,
    WeightsLike,
)

try:
    import ultralytics
except ImportError:
    raise ImportError("Please install 'ultralytics' first.")


current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region BASE CLASSES
# ==============================================================================

class Segmentor(ABC):
    """Base class for all segmentors."""

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        name: str,
        weights: WeightsLike | None = None,
        fg_color: Int3 = (255, 255, 255),
        bg_color: Int3 = (0, 0, 0),
        device: DeviceLike = "cpu",
        verbose: bool = True,
        *args, **kwargs
    ):
        """Initialize a new instance.

        Args:
            name (str): Name of the model variant.
            weights (WeightsLike, optional): Pre-trained weights to load.
                Defaults to None.
            fg_color (Int3, optional): Foreground color for masks.
                Defaults to (255, 255, 255).
            bg_color (Int3, optional): Background color for masks.
                Defaults to (0, 0, 0).
            device (DeviceLike, optional): Device to run the model on.
                Defaults to "cpu".
            verbose (bool, optional): Verbosity mode. Defaults to True.
        """
        super().__init__(name=name)

        # Assign attributes
        self.verbose = verbose
        self.fg_color = fg_color
        self.bg_color = bg_color
        self.device = sys_ctx.get_torch_device(device)

        # Allocate resources
        self._model: nn.Module | None = None

        self._init_model(name=name, weights=weights)

    @abstractmethod
    def _init_model(self, name: str, weights: WeightsLike | None = None):
        """Initialize the model with pre-trained weights."""
        pass

    # --- Properties ---
    @property
    def model(self) -> nn.Module:
        """Return the underlying model."""
        return self._model

    # --- Callable & Context Manager ---
    def __call__(self, *args, **kwargs):
        """Delegate the call to the underlying model."""
        return self.process(*args, **kwargs)

    @abstractmethod
    def process(self, *args, **kwargs) -> Any:
        """Process the input and return the segmentation mask."""
        pass

# endregion


# ==============================================================================
# region CONCRETE IMPLEMENTATIONS
# ==============================================================================

class SAMBoxSegmentor(Segmentor):
    """Segmentor for extracting foreground masks inside bounding boxes using
    Ultralytics SAM.

    References:
        - Code: https://docs.ultralytics.com/models/sam-2/
    """

    # --- Lifecycle & Initialization ---
    @override
    def _init_model(self, name: str, weights: WeightsLike | None = None):
        """Initialize the model with pre-trained weights."""
        if MODELS.has_model(name=name):
            model = MODELS.build(name=name, weights=weights, verbose=self.verbose)
            model = model.to(self.device)
            model.eval()
            self._model = model
        else:
            raise ValueError(f"Unsupported segmentation model: {name}")

     # --- Callable & Context Manager ---
    @override
    def process(self, image: ndarray, bbox: ndarray, *args, **kwargs) -> list[ndarray]:
        """Process the input image with bounding boxes.

        Args:
            image (ndarray): Input image array of shape (H, W, C) and pixel
                values ranging from 0 to 255.
            bbox (ndarray): Bounding boxes array of shape (N, 4+) and in CXCYWHN
                format.

        Returns:
            list[ndarray]: A list of binary masks for each bounding box.
        """
        if bbox.ndim != 2 or bbox.shape[-1] < 4:
            raise ValueError(
                f"Expected 'bbox' to be a 2D array of shape (N, 4), "
                f"but got {bbox.shape}."
            )

        # Convert bbox to XYXY format if necessary
        imgsz = Size.from_value(image)
        bbox = BBoxes(bbox=bbox, imgsz=imgsz).xyxy()

        # Extract masks using SAM
        results = self.model(image, bboxes=bbox[:, 0:4], device=self.device, verbose=False)
        results = results[0]

        # Create semantic mask
        semantic = np.zeros(image.shape, dtype=np.uint8)
        if results.masks is not None:
            for m in results.masks.xy:
                semantic = cv2.fillPoly(semantic, [np.array(m, dtype=np.int32)], self.fg_color)

        # Create foreground masks
        masks  = []
        kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.uint8)
        for b in bbox:
            x1, y1, x2, y2 = b.astype(int)
            m = semantic[y1:y2, x1:x2]
            # Dilate the mask to ensure it covers the background below the object
            m = cv2.dilate(m, kernel, iterations=1)
            masks.append(m)

        return masks

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
