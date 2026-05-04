#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Drawing Utilities.

This module provides utility functions for drawing shapes, text, and annotations
on images using OpenCV.
"""

from __future__ import annotations

__all__ = [
    "draw_info",
    "vis_heatmap",
]

import cv2
import matplotlib
import numpy as np
from box import Box
from matplotlib.colors import Colormap
from numpy import ndarray

from mon.core import Int2


# ====================================O==========================================
# region VISUALIZATION
# ==============================================================================

def vis_heatmap(image: ndarray, colormap: str | Colormap = "Spectral_r") -> ndarray:
    """Visualize a grayscale image as a heatmap using a specified colormap.

    Args:
        image (ndarray): Image array of shape (H, W) or (H, W, 1) and values
            ranging from 0 to 255, representing the grayscale intensity.
        colormap (str | Colormap, optional): Colormap to apply. Can be a string
            recognized by Matplotlib or a Colormap object. Defaults to "Spectral_r".

    Returns:
        ndarray: Image with the heatmap applied, of shape (H, W, 3) and values
            ranging from 0 to 255.
    """
    # 1. Handle dimensionality: Convert (H, W, 1) -> (H, W)
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image.squeeze(-1)
    elif image.ndim != 2:
        raise ValueError(
            f"Expected a 2D array or 3D array with 1 channel, "
            f"but got shape {image.shape}."
        )

    # 2. Safely normalize to [0.0, 1.0] for Matplotlib
    # If the image is uint8 or has values > 1.0, we assume it is scaled to 255
    if image.dtype == np.uint8 or image.max() > 1.0:
        norm_img = image.astype(np.float32) / 255.0
    else:
        norm_img = image.astype(np.float32)

    # Note: If you want the heatmap to strictly stretch from the absolute min
    # to max of the current image (auto-contrast), you would use this instead:
    norm_img = (norm_img - norm_img.min()) / (norm_img.max() - norm_img.min() + 1e-8)

    # 3. Retrieve the Matplotlib colormap safely
    if isinstance(colormap, str):
        cmap = matplotlib.colormaps.get_cmap(colormap)
    else:
        cmap = colormap

    # 4. Apply the colormap
    # Matplotlib colormaps return an RGBA array of shape (H, W, 4) with floats
    # in [0.0, 1.0]
    heatmap_rgba = cmap(norm_img)

    # 5. Drop the Alpha channel -> (H, W, 3) and scale back to [0, 255] uint8
    heatmap_rgb = (heatmap_rgba[..., :3] * 255).astype(np.uint8)

    return cv2.cvtColor(heatmap_rgb, cv2.COLOR_RGB2BGR)

# endregion


# ====================================O==========================================
# region DRAWING
# ==============================================================================

def draw_info(
    image: ndarray,
    info: dict | list,
    pos: Int2 = (20, 40),
    scale: float = 0.8,
    thickness: int = 2,
) -> ndarray:
    """Draw information on an image.

    Args:
        image (ndarray): Image array of shape (H, W, C) and values ranging from
            0 to 255.
        info (dict | list): Information to draw. Can be a dictionary
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
