#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Sobel Filter.

This module implements Sobel filter or Sobel operator.
"""

from __future__ import annotations

__all__ = [
    "sobel_filter",
]

import cv2
import numpy as np

from mon import core


def sobel_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Sobel filter.
    
    Args:
        image: An RGB image of type:
            - :obj:`torch.Tensor` in ``[B, C, H, W]`` format with data in
                the range ``[0.0, 1.0]``.
            - :obj:`numpy.ndarray` in ``[H, W, C]`` format with data in the
                range ``[0, 255]``.
        kernel_size: Size of the Sobel kernel. Default: ``3``.
    """
    if core.is_color_image(image):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Sobel filter in the x direction
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=kernel_size)
    # Sobel filter in the y direction
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=kernel_size)
    # Compute the magnitude of the gradient
    sobel_combined = cv2.magnitude(sobel_x, sobel_y)
    # Convert back to uint8
    sobel_combined = cv2.convertScaleAbs(sobel_combined)
    return sobel_combined
