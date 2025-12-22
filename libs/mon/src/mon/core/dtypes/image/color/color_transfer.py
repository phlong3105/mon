#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Color transfer function.

This module provides a function to transfer the color characteristics from a
source image to a target image using statistical methods in the LAB color space.

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
    """Transfer the color distribution from the target image to the source
    image using the mean and standard deviation of the LAB color space.

    Args:
        source (numpy.ndarray): An RGB source image as a numpy.ndarray of shape
            (H, W, 3) with pixel values in the range [0, 255].
        target (numpy.ndarray): An RGB target image as a numpy.ndarray of shape
            (H, W, 3) with pixel values in the range [0, 255].

    Returns:
        The color transferred image.
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
