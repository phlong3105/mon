#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn.functional import mse_loss as mse

__all__ = [
    "psnr",
]


def psnr(input: Tensor, target: Tensor, max_val: float) -> Tensor:
    """Create a function that calculates the PSNR between 2 images. PSNR is
    Peek Signal to Noise Ratio, which is similar to mean squared error.
    Given an m x n image, the PSNR is:

    .. math::
        \text{PSNR} = 10 \log_{10} \bigg(\frac{\text{MAX}_I^2}{MSE(I,T)}\bigg)

    where

    .. math::
        \text{MSE}(I,T) = \frac{1}{mn}\sum_{i=0}^{m-1}\sum_{j=0}^{n-1} [I(i,
        j) - T(i,j)]^2

    and :math:`\text{MAX}_I` is the maximum possible input value
    (e.g for floating point images :math:`\text{MAX}_I=1`).

    Reference:
        https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio#Definition
        
    Args:
        input (Tensor):
            Input image with arbitrary shape [*].
        target (Tensor):
            Labels image with arbitrary shape [*].
        max_val (float):
            Maximum value in the input image.

    Return:
        (Tensor):
            Computed loss as a scalar.

    Examples:
        >>> ones = torch.ones(1)
        >>> psnr(ones, 1.2 * ones, 2.) # 10 * log(4/((1.2-1)**2)) / log(10)
        image(20.0000)
    """
    if not isinstance(input, Tensor):
        raise TypeError(f"Expected Tensor but got: {type(input)}.")
    if not isinstance(target, Tensor):
        raise TypeError(f"Expected Tensor but got: {type(target)}.")
    if input.shape != target.shape:
        raise TypeError(f"Expected tensors of equal shapes, but "
                        f"got: {input.shape} and {target.shape}")

    return 10.0 * torch.log10(
        max_val ** 2 / mse(input, target, reduction="mean")
    )
