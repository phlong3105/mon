#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements basic image adjustment functions."""

from __future__ import annotations

__all__ = [
    "adjust_gamma",
]

import cv2
import numpy as np
import torch

from mon.nn import functional as F


# region Adjust

def adjust_gamma(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Adjust gamma value in the image using the Power Law Transform.
    
    First, our image pixel intensities must be scaled from the range
    :math:`[0, 255]` to :math:`[0, 1.0]`. From there, we obtain our output gamma
    corrected image by applying the following equation:
    .. math::
        O = I ^ {(1 / G)}
    Where I is our input image and G is our gamma value. The output image ``O``
    is then scaled back to the range :math:`[0, 255]`.
    
    Args:
        image: An image.
        gamma: A gamma correction value
            - < 1 will make the image darker.
            - > 1 will make the image lighter.
            - = 1 will have no effect on the input image.
        
    Returns:
        A gamma-corrected image.
    """
    # Build a lookup table mapping the pixel values [0, 255] to their adjusted
    # gamma values.
    inv_gamma = 1.0 / gamma
    table     = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)])
    table.astype("uint8")
    # Apply gamma correction using the lookup table
    return cv2.LUT(image, table)

# endregion
