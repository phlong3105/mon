#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Depth visualization operations.

This module provides functions to visualize the depth data for debugging and
human interaction.
"""

__all__ = [
    "to_color",
]

import cv2
import numpy as np

from .. import image as I


# ==============================================================================
# CANVAS CONFIGURATION (Canvas Setup)
# ==============================================================================

# --- Style (Colormaps, Palettes, Themes) ---


# ==============================================================================
# RENDERING ENGINES (Drawing logic)
# ==============================================================================

# --- Decorate (Drawing Overlays, BBoxes, Text) ---
def to_color(depth: np.ndarray, color_map: int = cv2.COLORMAP_JET) -> np.ndarray:
    """Convert a depth map to a color-coded image.

    Args:
        depth: A depth map, formatted as a numpy array of shape (H, W) and pixel
            values ranging from 0.0 to 1.0 or in absolute depth units (e.g., mm).
        color_map: OpenCV colormap to use for coloring. Defaults to cv2.COLORMAP_JET.

    Returns:
        Color-coded depth image, formatted as a numpy.ndarray with shape (H, W, 3)
        and pixel values ranging from 0 to 255.

    Raises:
        TypeError: If ``depth`` is not a numpy.ndarray.
    """
    if not isinstance(depth, np.ndarray):
        raise TypeError(f"``depth`` must be a numpy.ndarray, got {type(depth)}.")
    depth = np.uint8(255 * depth) if I.is_normalized(depth) else depth
    depth = cv2.applyColorMap(depth, color_map)
    return depth


# --- Compose (Creating Grids and Collages) ---


# ==============================================================================
# DISPLAY & PLOTTING (High-level wrappers)
# ==============================================================================

# --- Render (Notebook/GUI display logic) ---


# --- Snapshot (Saving visual previews for QA) ---
