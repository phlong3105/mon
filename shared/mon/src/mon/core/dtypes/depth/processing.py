#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for depth map processing functions.

This module provides utility functions for processing depth maps, including
conversion to color-coded images.
"""

__all__ = [
    "to_color",
]

import cv2
import numpy as np

from .. import image as I


# ----- Conversion -----
def to_color(depth: np.ndarray, color_map: int = cv2.COLORMAP_JET) -> np.ndarray:
    """Converts a depth map to a color-coded image using a specified colormap.
    
    Args:
        depth (numpy.ndarray): The input depth map as a 2D array.
        color_map (int): The OpenCV colormap to use for conversion. Defaults to
            cv2.COLORMAP_JET.
            
    Returns:
        numpy.ndarray: The color-coded depth image.
    
    Raises:
        TypeError: If ``depth`` is not a numpy.ndarray.
    """
    if not isinstance(depth, np.ndarray):
        raise TypeError(f"``depth`` must be a numpy.ndarray, got {type(depth)}.")
    depth = np.uint8(255 * depth) if I.is_normalized(depth) else depth
    depth = cv2.applyColorMap(depth, color_map)
    return depth
