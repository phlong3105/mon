#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for sobel filters.

This module implements the Sobel filter for edge detection in images.
"""

__all__ = [
    "sobel_filter",
]

import cv2
import numpy as np

from ..utils import is_color


# ----- Sobel Filter -----
def sobel_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Applies Sobel filter to detect edges in an image.

    Args:
        image (numpy.ndarray): Input image as a numpy.ndarray of shape (H, W)
            for grayscale or (H, W, C) for color images with pixel values in
            [0, 255].
        kernel_size (int): Size of the Sobel kernel. Must be odd and greater
            than 1. Defaults to 3.
            
    Returns:
        numpy.ndarray: Image after applying Sobel filter, of the same shape
            as the input image.
    
    Raises:
        TypeError: If ``image`` is not a numpy.ndarray with 2 or 3 dimensions.
    """
    if not isinstance(image, np.ndarray) or image.ndim not in [2, 3]:
        raise TypeError(f"``image`` must be a numpy.ndarray with 2 or 3 dimensions, "
                        f"got {type(image)} with shape {image.shape}.")
    
    if is_color(image):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=kernel_size)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=kernel_size)
    sobel_combined = cv2.magnitude(sobel_x, sobel_y)
    sobel_combined = cv2.convertScaleAbs(sobel_combined)
    return sobel_combined
