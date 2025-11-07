#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements processing function for depth data type.

Common Tasks:
    - Format conversions.
"""

__all__ = [
    "to_color",
]

import cv2
import numpy as np

from mon.core.dtypes import image as I


# ----- Conversion -----
def to_color(depth: np.ndarray, color_map: int = cv2.COLORMAP_JET) -> np.ndarray:
    """Converts a depth map to a color-coded image.

    Args:
        depth: Depth map as a ``numpy.ndarray`` of shape :math:`(H, W, 1)`.
        color_map: Color map for the depth map. Default: ``cv2.COLORMAP_JET``.
        use_rgb: Convert to RGB format if ``True``. Default: ``False``.
    
    Returns:
        Color-coded depth map as a ``numpy.ndarray`` of shape :math:`(H, W, 3)`.
    
    Raises:
        TypeError: If ``depth`` is not a ``numpy.ndarray``.
    """
    if not isinstance(depth, np.ndarray):
        raise TypeError(f"``depth`` must be a numpy.ndarray, got {type(depth)}.")
    depth = np.uint8(255 * depth) if I.is_normalized(depth) else depth
    depth = cv2.applyColorMap(depth, color_map)
    return depth
