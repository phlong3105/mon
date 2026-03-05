#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Drawing Utilities.

This module provides utility functions for drawing shapes, text, and annotations
on images using OpenCV.
"""

from __future__ import annotations

__all__ = [
    "draw_info",
]

import cv2
from box import Box
from numpy import ndarray

from mon.core import DictLike, Int2


# ====================================O==========================================
# region VISUALIZATION
# ==============================================================================

def draw_info(
    image: ndarray,
    info: DictLike | list,
    pos: Int2 = (20, 40),
    scale: float = 0.8,
    thickness: int = 2,
) -> ndarray:
    """Draw information on an image.

    Args:
        image (ndarray): Image array of shape (H, W, C) and values ranging from
            0 to 255.
        info (DictLike | list): Information to draw. Can be a dictionary
            (key-value pairs) or a list of strings.
        pos (Int2, optional): Position (x, y) where the information will be
            drawn. Defaults to (20, 40).
        scale (float, optional): Font scale for the text. Defaults to 0.8.
        thickness (int, optional): Thickness of the text. Defaults to 2.

    Returns:
        ndarray: Image with the drawn information.
    """
    if isinstance(info, (Box, dict)):
        lines = [f"{k}: {v}" for k, v in info.items()]
    else:
        lines = info
    if not isinstance(info, list):
        raise TypeError(
            f"Expected a list or dictionary, but got {type(info).__name__}."
        )

    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y = pos
    line_spacing = int(40 * scale)

    image = image.copy()
    for i, line in enumerate(lines):
        current_y = y + (i * line_spacing)
        cv2.putText(image, line, (x, current_y), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(image, line, (x, current_y), font, scale, (0, 255, 0), thickness, cv2.LINE_AA)

    return image

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
