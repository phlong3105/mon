#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Ultralytics SAM models and pre-trained weights.

This module provides wrappers for Ultralytics SAM models and pre-trained weights.

References:
    - Code: https://github.com/ultralytics/ultralytics
"""

from __future__ import annotations

__all__ = [

]

import numpy as np
import torch

from mon.core import create_device, MODELS, Path, Task
from mon.core.dtypes import WeightsEnum

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


# --- Mixins ---


# endregion


# ==============================================================================
# region CONCRETE IMPLEMENTATIONS
# ==============================================================================

class BoxSegmentor:
    """Segmentor for extracting foreground masks inside bounding boxes."""

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        name    : str,
        weights : WeightsEnum | None       = None,
        device  : torch.device | str | int = "cpu",
        fg_color: tuple[int, int, int]     = (255, 255, 255),
        bg_color: tuple[int, int, int]     = (0, 0, 0),
        verbose : bool                     = False,
        *args, **kwargs
    ):
        """Initialize a new instance.

        Args:
            name: Name of the segmentation model.
            weights: Pre-trained weights to load. Defaults to None.
            device: Device to use for inference. Defaults to "cpu".
            fg_color: Color for the foreground mask. Defaults to white.
            bg_color: Color for the background mask. Defaults to black.
            verbose: If True, enable verbose output. Defaults to False.
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
            self.model = MODELS.build(name=name, weights=weights, *args, **kwargs)
        else:
            raise ValueError(f"Unsupported segmentation model: {name}")

    # --- Callable & Context Manager ---
    def __call__(self, image: np.ndarray, labels: np.ndarray, *args, **kwargs):
        return self._process(image=image, labels=labels, *args, **kwargs)

    def _process(self, image: np.ndarray, labels: np.ndarray, *args, **kwargs):
        pass

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
