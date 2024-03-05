#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements basic image adjustment functions."""

from __future__ import annotations

__all__ = [
    "add_weighted",
    "adjust_gamma",
    "blend",
]

import cv2
import multipledispatch
import numpy as np
import torch
from torchvision.transforms import functional as TF


# region Blend

def add_weighted(
    input1: torch.Tensor | np.ndarray,
    alpha : float,
    input2: torch.Tensor | np.ndarray,
    beta  : float,
    gamma : float = 0.0,
) -> torch.Tensor | np.ndarray:
    """Calculate the weighted sum of two image tensors as follows:
        output = image1 * alpha + image2 * beta + gamma

    Args:
        input1: The first image.
        alpha: The weight of the :param:`image1` elements.
        input2: The second image.
        beta: The weight of the :param:`image2` elements.
        gamma: A scalar added to each sum. Default: ``0.0``.

    Returns:
        A weighted image.
    """
    if input1.shape != input2.shape:
        raise ValueError(
            f"The shape of x and y must be the same, but got "
            f"{input1.shape} and {input2.shape}."
        )
    bound  = 1.0 if input1.is_floating_point() else 255.0
    output = input1 * alpha + input2 * beta + gamma
    if isinstance(output, torch.Tensor):
        output = output.clamp(0, bound).to(input1.dtype)
    elif isinstance(output, np.ndarray):
        output = np.clip(output, 0, bound)
    else:
        raise TypeError(
            f":param:`input` must be a :class:`np.ndarray` or :class:`torch.Tensor`, "
            f"but got {type(input)}."
        )
    return output


def blend(
    input1: torch.Tensor | np.ndarray,
    input2: torch.Tensor | np.ndarray,
    alpha : float,
    gamma : float = 0.0
) -> torch.Tensor | np.ndarray:
    """Blend 2 images together using the formula:
        output = :param:`image1` * alpha + :param:`image2` * beta + gamma

    Args:
        input1: A source image.
        input2: A n overlay image that we want to blend on top of
            :param:`image1`.
        alpha: An alpha transparency of the overlay.
        gamma: A scalar added to each sum. Default: ``0.0``.

    Returns:
        A blended image.
    """
    return add_weighted(
        input1 = input2,
        alpha  = alpha,
        input2 = input1,
        beta   = 1.0 - alpha,
        gamma  = gamma,
    )

# endregion


# region Gamma

@multipledispatch.dispatch(torch.Tensor)
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


@multipledispatch.dispatch(np.ndarray)
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
    # Build a lookup table mapping the pixel values [0, 255] to their adjusted
    # gamma values.
    inv_gamma = 1.0 / gamma
    table     = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)])
    table.astype("uint8")
    # Apply gamma correction using the lookup table
    return cv2.LUT(image, table)

# endregion
