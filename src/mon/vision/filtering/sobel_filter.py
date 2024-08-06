#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements Sobel filter or Sobel operator."""

from __future__ import annotations

__all__ = [
    "sobel_filter",
]

import cv2
import numpy as np
from plum import dispatch


@dispatch
def sobel_filter(input: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Sobel filter."""
    h, w, c = input.shape
    if c == 3:
        input = cv2.cvtColor(input, cv2.COLOR_RGB2GRAY)
    # Sobel filter in the x direction
    sobel_x = cv2.Sobel(input, cv2.CV_64F, 1, 0, ksize=kernel_size)
    # Sobel filter in the y direction
    sobel_y = cv2.Sobel(input, cv2.CV_64F, 0, 1, ksize=kernel_size)
    # Compute the magnitude of the gradient
    sobel_combined = cv2.magnitude(sobel_x, sobel_y)
    # Convert back to uint8
    sobel_combined = cv2.convertScaleAbs(sobel_combined)
    return sobel_combined
