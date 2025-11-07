#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements color transfer between images.

References:
    - Paper: "Color Transfer between Images".
    - Code: https://github.com/rinsa318/color-transfer
    - Code: https://github.com/chia56028/Color-Transfer-between-Images
    - Code: https://www.cnblogs.com/likethanlove/p/6003677.html
    - Code: https://pyimagesearch.com/2014/06/30/super-fast-color-transfer-images/
"""

__all__ = [
    "color_transfer",
]

import cv2
import numpy as np


def color_transfer(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Transfer color from source image to target image.

    Args:
        source: Source image as a ``numpy.ndarray`` of shape :math:`(H, W, 3)`
            in range :math:`[0, 255]` and RGB format.
        target: Same as type and format as `source`.

    Returns:
        The color transferred image as a ``numpy.ndarray`` of shape :math:`(H, W, 3)`
        in range :math:`[0, 255]` and RGB format.
    """
    # Convert to LAB color space
    s = cv2.cvtColor(source, cv2.COLOR_RGB2LAB).astype(np.float32)
    t = cv2.cvtColor(target, cv2.COLOR_RGB2LAB).astype(np.float32)

    # Compute mean and std for each channel
    s_mean = np.mean(s, axis=(0, 1))
    s_std  = np.std(s,  axis=(0, 1))
    t_mean = np.mean(t, axis=(0, 1))
    t_std  = np.std(t,  axis=(0, 1))

    # Apply color transfer using vectorized operations
    s = (s - s_mean) * (t_std / np.maximum(s_std, 1e-10)) + t_mean

    # Clip values to valid range and convert to uint8
    s = np.clip(np.round(s), 0, 255).astype("uint8")

    # Convert back to RGB
    return cv2.cvtColor(s, cv2.COLOR_LAB2RGB)
