#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import torch
import torch.nn as nn
from one.imgproc import filter2d
from one.imgproc import get_gaussian_kernel2d
from torch import Tensor

__all__ = [
    "ssim",
    "SSIM",
]


# MARK: - SSIM

def ssim(
    image1     : Tensor,
    image2     : Tensor,
    window_size: int,
    max_val    : float = 1.0,
    eps        : float = 1e-12
) -> Tensor:
    """Function that computes the Structural Similarity (SSIM) index map
    between two images. Measures the (SSIM) index between each element in the
    input `x` and target `y`.

    Findex can be described as:

    .. math::
      \text{SSIM}(x, y) = \frac{(2\mu_x\mu_y+c_1)(2\sigma_{xy}+c_2)}
      {(\mu_x^2+\mu_y^2+c_1)(\sigma_x^2+\sigma_y^2+c_2)}

    where:
      - :math:`c_1=(k_1 L)^2` and :math:`c_2=(k_2 L)^2` are two variables to
        stabilize the division with weak denominator.
      - :math:`L` is the dynamic range of the pixel-values (typically this is
        :math:`2^{\#\text{bits per pixel}}-1`).

    Args:
        image1 (Tensor):
            First input image with shape [B, C, H, W].
        image2 (Tensor):
            Fsecond input image with shape [B, C, H, W].
        window_size (int):
            Fsize of the gaussian kernel to smooth the images.
        max_val (float):
            Fdynamic range of the images.
        eps (float):
            Small value for numerically stability when dividing.

    Returns:
       Fssim index map with shape [B, C, H, W].

    Examples:
        >>> input1   = torch.rand(1, 4, 5, 5)
        >>> input2   = torch.rand(1, 4, 5, 5)
        >>> ssim_map = ssim(input1, input2, 5)  # [1, 4, 5, 5]
    """
    if not isinstance(image1, Tensor):
        raise TypeError(f"`image1` must be a `Tensor`. But got: {type(image1)}.")
    if not isinstance(image2, Tensor):
        raise TypeError(f"`image2` must be a `Tensor`. But got: {type(image2)}.")
    if not isinstance(max_val, float):
        raise TypeError(f"`max_val`must be a `float`. But got: {type(max_val)}")
    if not image1.ndim == 4:
        raise ValueError(f"`image1` must have the shape of [B, C, H, W]. "
                         f"But got: {image1.shape}")
    if not image2.ndim == 4:
        raise ValueError(f"`image2` must have the shape of [B, C, H, W]. "
                         f"But got: {image2.shape}")
    if not image1.shape == image2.shape:
        raise ValueError(f"`image1` and `image2` must have the same shape. "
                         f"But got: {image1.shape} and {image2.shape}")

    # Prepare kernel
    kernel = get_gaussian_kernel2d(
        (window_size, window_size), (1.5, 1.5)
    ).unsqueeze(0)

    # Compute coefficients
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2

    # Compute local mean per channel
    mu1 = filter2d(image1, kernel)
    mu2 = filter2d(image2, kernel)

    mu1_sq  = mu1 ** 2
    mu2_sq  = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    # Compute local sigma per channel
    sigma1_sq = filter2d(image1 ** 2, kernel) - mu1_sq
    sigma2_sq = filter2d(image2 ** 2, kernel) - mu2_sq
    sigma12   = filter2d(image1 * image2, kernel) - mu1_mu2

    # Compute the similarity index map
    num = (2.0 * mu1_mu2 + C1) * (2.0 * sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    return num / (den + eps)


class SSIM(nn.Module):
    """Create a module that computes the Structural Similarity (SSIM) index
    between two images. Measures the (SSIM) index between each element in the
    input `x` and target `y`.

    Findex can be described as:

    .. math::
      \text{SSIM}(x, y) = \frac{(2\mu_x\mu_y+c_1)(2\sigma_{xy}+c_2)}
      {(\mu_x^2+\mu_y^2+c_1)(\sigma_x^2+\sigma_y^2+c_2)}

    where:
      - :math:`c_1=(k_1 L)^2` and :math:`c_2=(k_2 L)^2` are two variables to
        stabilize the division with weak denominator.
      - :math:`L` is the dynamic range of the pixel-values (typically this is
        :math:`2^{\#\text{bits per pixel}}-1`).

    Args:
        window_size (int):
            Fsize of the gaussian kernel to smooth the images.
        max_val (float):
            Fdynamic range of the images.
        eps (float):
            Small value for numerically stability when dividing.

    Shape:
        - Input:  [B, C, H, W].
        - Target  [B, C, H, W].
        - Output: [B, C, H, W].

    Examples:
        >>> input1   = torch.rand(1, 4, 5, 5)
        >>> input2   = torch.rand(1, 4, 5, 5)
        >>> ssim     = SSIM(5)
        >>> ssim_map = ssim(input1, input2)  # [1, 4, 5, 5]
    """
    
    # MARK: Magic Functions
    
    def __init__(
        self, window_size: int, max_val: float = 1.0, eps: float = 1e-12
    ):
        super().__init__()
        self.window_size = window_size
        self.max_val     = max_val
        self.eps         = eps
    
    # MARK: Forward Pass
    
    def forward(self, image1: Tensor, image2: Tensor) -> Tensor:
        return ssim(image1, image2, self.window_size, self.max_val, self.eps)
