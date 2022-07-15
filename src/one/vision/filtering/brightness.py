#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import cv2
import numpy as np
from torch import Tensor
from torchvision.transforms import functional_pil as F_pil
from torchvision.transforms import functional_tensor as F_t

from one.core import denormalize_naive
from one.core import is_normalized
from one.core import normalize_naive
from one.core import TensorOrArray

__all__ = [
    "adjust_gamma",
]


# MARK: - Functional

def adjust_gamma_tensor(image: Tensor, gamma: float, gain: float = 1.0) -> Tensor:
    r"""Perform gamma correction on an image. Also known as Power Law Transform.
    Intensities in RGB mode are adjusted based on the following equation:

    .. math::
        I_{\text{out}} = 255 \times \text{gain} \times \left(\frac{I_{\text{in}}}{255}\right)^{\gamma}

    References:
        https://en.wikipedia.org/wiki/Gamma_correction

    Args:
        image (Tensor[..., 1 or 3, H, W]):
            PIL Image to be adjusted.
            If img is torch Tensor, it is expected to be in [..., 1 or 3, H, W]
            format, where ... means it can have an  arbitrary number of leading
            dimensions.
            If img is PIL Image, modes with transparency (alpha channel) are not
            supported.
        gamma (float):
            Non-negative real number, same as :math:`\gamma` in the equation.
            `gamma` larger than 1 make the shadows darker, while gamma smaller
            than 1 make dark regions lighter.
        gain (float):
            The constant multiplier.
    
    Returns:
        img (Tensor[..., 1 or 3, H, W]):
            Gamma correction adjusted image.
    """
    if not isinstance(image, Tensor):
        return F_pil.adjust_gamma(image, gamma, gain)

    img = normalize_naive(image)
    img = F_t.adjust_gamma(img, gamma, gain)
    if is_normalized(image):
        return img
    else:
        return denormalize_naive(img)


def adjust_gamma_numpy(image: np.ndarray, gamma: float = 1.0, gain: float = 1.0) -> np.ndarray:
    r"""Perform gamma correction on an image. Also known as Power Law Transform.
    Intensities in RGB mode are adjusted based on the following equation:

    .. math::
        I_{\text{out}} = 255 \times \text{gain} \times \left(\frac{I_{\text{in}}}{255}\right)^{\gamma}

    References:
        https://en.wikipedia.org/wiki/Gamma_correction

    Args:
        image (np.ndarray[..., 1 or 3, H, W]):
            PIL Image to be adjusted.
            If img is torch Tensor, it is expected to be in [..., 1 or 3, H, W]
            format, where ... means it can have an  arbitrary number of leading
            dimensions.
            If img is PIL Image, modes with transparency (alpha channel) are not
            supported.
        gamma (float):
            Non-negative real number, same as :math:`\gamma` in the equation.
            `gamma` larger than 1 make the shadows darker, while gamma smaller
            than 1 make dark regions lighter.
        gain (float):
            The constant multiplier.
    
    Returns:
        img (np.ndarray[..., 1 or 3, H, W]):
            Gamma correction adjusted image.
    """
    if is_normalized(image):
        img = denormalize_naive(image)
    else:
        img = image.copy()
    
    inv_gamma = gamma
    table     = np.array([(gain * (i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    img       = cv2.LUT(img, table)
    if is_normalized(image):
        return img
    else:
        return img
    

def adjust_gamma(image: TensorOrArray, gamma: float = 1.0, gain: float = 1.0) -> TensorOrArray:
    if isinstance(image, Tensor):
        return adjust_gamma_tensor(image=image, gamma=gamma, gain=gain)
    elif isinstance(image, np.ndarray):
        return adjust_gamma_numpy(image=image, gamma=gamma, gain=gain)
    else:
        raise TypeError(f"Do not support: {type(image)}.")
