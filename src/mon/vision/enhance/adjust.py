#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements basic image adjustment functions."""

from __future__ import annotations

__all__ = [
    "add_noise",
    "adjust_gamma",
]

from typing import Literal

import cv2
import numpy as np
import torch
from plum import dispatch
from torchvision.transforms import functional as TF


# region Gamma

@dispatch
def adjust_gamma(image: torch.Tensor, gamma: float = 1.0, gain: float = 1.0) -> torch.Tensor:
    """Adjust gamma value in the image. Also known as Power Law Transform.
    
    Intensities in RGB mode are adjusted based on the following equation:
    
    .. math::
        I_{\text{out}} = 255 \times \text{gain} \times \left(\frac{I_{\text{in}}}{255}\right)^{\gamma}
    
    Args:
        image: An image in :math:`[B, C, H, W]` format.
        gamma: Non-negative real number, same as :math:`\gamma` in the equation.
            :param:`gamma` larger than ``1`` makes the shadows darker, while
            :param:`gamma` smaller than ``1`` makes dark regions lighter.
        gain: The constant multiplier.
        
    Returns:
        A gamma-corrected image.
    """
    return TF.adjust_gamma(img=image, gamma=gamma, gain=gain)


@dispatch
def adjust_gamma(image: np.ndarray, gamma: float = 1.0, gain: float = 1.0) -> np.ndarray:
    """Adjust gamma value in the image. Also known as Power Law Transform.
    
    First, our image pixel intensities must be scaled from the range
    :math:`[0, 255]` to :math:`[0, 1.0]`. From there, we obtain our output gamma
    corrected image by applying the following equation:
    .. math::
        O = I ^ {(1 / G)}
    Where I is our input image and G is our gamma value. The output image ``O``
    is then scaled back to the range :math:`[0, 255]`.
    
    Args:
        image: An image in :math:`[H, W, C]` format.
        gamma: A gamma correction value
            - < 1 will make the image darker.
            - > 1 will make the image lighter.
            - = 1 will have no effect on the input image.
        gain: The constant multiplier.
        
    Returns:
        A gamma-corrected image.
    """
    # Build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values.
    inv_gamma = 1.0 / gamma
    table     = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)])
    table.astype("uint8")
    # Apply gamma correction using the lookup table
    return cv2.LUT(image, table)

# endregion


# region Noise

def add_noise(
    image      : torch.Tensor,
    noise_level: int = 25,
    noise_type : Literal["gaussian", "poisson"] = "gaussian"
) -> torch.Tensor:
    """Add noise to an image.
    
    Args:
        image: The input image.
        noise_level: The noise level.
        noise_type: The type of noise to add: ``'gaussian'`` or ``'poisson'``.
            Default: ``"gaussian"``.
        
    Returns:
        A noisy image.
    """
    if noise_type == "gaussian":
        noisy = image + torch.normal(0, noise_level / 255, image.shape)
        noisy = torch.clamp(noisy, 0, 1)
    elif noise_type == "poisson":
        noisy = torch.poisson(noise_level * image) / noise_level
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    return noisy

# endregion
