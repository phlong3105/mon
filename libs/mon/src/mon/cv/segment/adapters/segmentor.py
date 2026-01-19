#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Ultralytics SAM models and pre-trained weights.

This module provides wrappers for segmentation models to extract masks.
"""

from __future__ import annotations

__all__ = [
    "BoxSegmentor",
    "SAMBoxSegmentor",
]

import abc

import cv2
import numpy as np
import torch

from mon.core import (
    BBoxFormat,
    create_device,
    create_progress_bar,
    EXT,
    MODELS,
    Path,
    Task,
)
from mon.core.dtypes import bbox as B, image as I, WeightsType

try:
    import ultralytics
except ImportError:
    raise ImportError("Please install 'ultralytics' first.")


current_file = Path(__file__).normalize()
current_dir  = current_file.parents[0]


# ==============================================================================
# region BASE CLASSES & MIXINS
# ==============================================================================

# --- Base Classes ---

class BoxSegmentor(abc.ABC):
    """Segmentor for extracting foreground masks inside bounding boxes."""

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        name    : str,
        weights : WeightsType | None       = None,
        device  : torch.device | str | int = "cpu",
        fg_color: tuple[int, int, int]     = (255, 255, 255),
        bg_color: tuple[int, int, int]     = (0, 0, 0),
        verbose : bool                     = True,
        *args, **kwargs
    ):
        """Initialize a new instance.

        Args:
            name: Name of the segmentation model.
            weights: Pre-trained weights to load. Defaults to None.
            device: Device to use for inference. Defaults to "cpu".
            fg_color: Color for the foreground mask. Defaults to white.
            bg_color: Color for the background mask. Defaults to black.
            verbose: Verbosity mode. Defaults to True.
            *args: Additional positional arguments for the model.
            **kwargs: Additional keyword arguments for the model.
        """
        super().__init__(*args, **kwargs)
        self.verbose  = verbose
        self.device   = create_device(device)
        self.fg_color = fg_color
        self.bg_color = bg_color

        # Build the segmentation model
        if MODELS.has(name=name, task=Task.SEGMENT):
            model       = MODELS.build(name=name, weights=weights, verbose=verbose, *args, **kwargs)
            model       = model.to(self.device)
            self._model = model
        else:
            raise ValueError(f"Unsupported segmentation model: {name}")

    # --- Callable & Context Manager ---
    def __call__(self, image: np.ndarray, bbox: np.ndarray, *args, **kwargs):
        """Delegate the call to the underlying model."""
        return self.process(image=image, bbox=bbox)

    def progress_dir(self, *args, **kwargs):
        """Process a single image directory."""
        raise NotImplementedError(f"This method is not implemented for {self.__class__.__name__}.")

    @abc.abstractmethod
    def process(self, image: np.ndarray, bbox: np.ndarray) -> list[np.ndarray]:
        """Process the input image with bounding boxes.

        Args:
            image: Input image, formatted as a numpy ndarray of shape (H, W, C)
                and pixel values ranging from 0 to 255.
            bbox: Bounding boxes, formatted as a numpy.ndarray of shape (N, 4+)
                and in XYXY or CXCYWHN format.

        Returns:
            A list of numpy.ndarray representing the extracted masks for each
            bounding box.
        """
        pass


# --- Mixins ---


# endregion


# ==============================================================================
# region CONCRETE IMPLEMENTATIONS
# ==============================================================================

class SAMBoxSegmentor(BoxSegmentor):
    """Segmentor for extracting foreground masks inside bounding boxes using
    Ultralytics SAM.

    References:
        - Code: https://docs.ultralytics.com/models/sam-2/
    """

    # --- Lifecycle & Initialization ---
    def __init__(self, name: str, *args, **kwargs):
        """Initialize a new instance.

        Args:
            name: Name of the segmentation model.
            *args: Additional positional arguments for the model.
            **kwargs: Additional keyword arguments for the model.
        """
        if not MODELS.has(name=name, task=Task.SEGMENT, arch="sam"):
            raise ValueError(f"Unsupported segmentation model: {name}")

        super().__init__(name=name, *args, **kwargs)

    # --- Callable & Context Manager ---
    def process(self, image: np.ndarray, bbox: np.ndarray) -> list[np.ndarray]:
        """Process the input image with bounding boxes.

        Args:
            image: Input image, formatted as a numpy ndarray of shape (H, W, C)
                and pixel values ranging from 0 to 255.
            bbox: Bounding boxes, formatted as a numpy.ndarray of shape (N, 4+)
                and in XYXY or CXCYWHN format.

        Returns:
            A list of numpy.ndarray representing the extracted masks for each
            bounding box.
        """
        if bbox.ndim != 2 or bbox.shape[-1] < 4:
            raise ValueError(
                f"Expected 'bbox' to be a 2D numpy.ndarray of shape [N, 4], "
                f"but got {bbox.shape}."
            )

        # Convert bbox to XYXY format if necessary
        imgsz = I.imgsz(image)
        if B.is_cxcywhn(bbox):
            bbox = B.convert(bbox, fmt=BBoxFormat.CXCYWHN2XYXY, imgsz=imgsz)

        if not B.is_xyxy(bbox):
            raise ValueError(f"Expected 'bbox' to be in XYXY format, but got {bbox.shape}.")

        # Extract masks using SAM
        results = self._model(image, bboxes=bbox[:, 0:4], device=self.device, verbose=False)
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
