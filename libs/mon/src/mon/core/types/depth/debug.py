#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Depth debugging utilities.

This module provides debugging utilities for depth data.
"""

from __future__ import annotations

__all__ = [
    "to_color",
]

import cv2
import numpy as np


# ==============================================================================
# region BASIC LOGGING
# ==============================================================================


# endregion


# ==============================================================================
# region VISUALIZATION
# ==============================================================================

def to_color(depth: np.ndarray, color_map: int = cv2.COLORMAP_JET) -> np.ndarray:
    """Convert a depth map to a color-coded image.

    Args:
        depth: Depth map, formatted as a numpy.ndarray with shape (H, W) or
            (H, W, 1) and pixel values ranging from 0.0 to 1.0 or in absolute
            depth units.
        color_map: OpenCV colormap to use for coloring. Defaults to
            cv2.COLORMAP_JET.

    Returns:
        Color-coded depth image, formatted as a numpy.ndarray with shape
        (H, W, 3) and pixel values ranging from 0 to 255.

    Raises:
        TypeError: If ``depth`` is not a numpy.ndarray.
    """
    if not isinstance(depth, np.ndarray):
        raise TypeError(f"Expected 'depth' to be a numpy.ndarray, but got {type(depth)}.")

    # Handle dimensionality (Ensure H, W)
    if depth.ndim == 3:
        depth = depth.squeeze()

    # Normalize to 0-255 (Standardizes contrast)
    # cv2.normalize is faster and handles min/max scaling efficiently
    depth_8bit  = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Apply Colormap
    color_depth = cv2.applyColorMap(depth_8bit, color_map)

    # OpenCV uses BGR by default, convert to RGB for standard library consistency
    return cv2.cvtColor(color_depth, cv2.COLOR_BGR2RGB)

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
