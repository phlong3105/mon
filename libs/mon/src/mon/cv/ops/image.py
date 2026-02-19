#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Image Operation.

This module provides operations for images.
"""

from __future__ import annotations

__all__ = [
    "pair_downsample",
]

import torch
from torch import Tensor
from torch.nn import functional as F


# ==============================================================================
# region VALIDATION
# ==============================================================================

# endregion


# ==============================================================================
# region RETRIEVAL
# ==============================================================================

# --- Accessing ---


# --- Selection ---


# --- Aggregation ---

# endregion


# ==============================================================================
# region MUTATION
# ==============================================================================

# --- Alternation ---


# --- Rearrangement ---


# --- Addition ---


# --- Removal ---

# endregion


# ==============================================================================
# region COMPUTATION
# ==============================================================================

# --- Arithmetic ---


# --- Comparison ---


# --- Logical ---


# --- Geometric ---

# endregion


# ==============================================================================
# region TRANSFORMATION
# ==============================================================================

# --- Casting ---


# --- Encoding ---


# --- Standardization ---


# --- Structural ---


# --- Statistical ---


# --- Geometric ---

def pair_downsample(image: Tensor) -> tuple[Tensor, Tensor]:
    """Downsample an image tensor into a pair to half resolution.

    References:
        - Code: https://colab.research.google.com/drive/1i82nyizTdszyHkaHBuKPbWnTzao8HF9b?usp=sharing

    Args:
        image (Tensor): Image tensor of shape (B, C, H, W) and values ranging
            from 0.0 to 1.0.

    Returns:
        tuple[Tensor, Tensor]: Downsampled images of shape (B, C, H/2, W/2).

    Raises:
        TypeError: If ``image`` is not a 4D torch.Tensor.

    Notes:
        Averages diagonal pixels in non-overlapping patches:
            -------------      -------------
            | A1 | B1 | A2 | B2 |      | A1+D1/2 | A2+D2/2 |
            | C1 | D1 | C2 | D2 |      | A3+D3/2 | A4+D4/2 |
            -------------  =>  -------------
            | A3 | B3 | A4 | B4 |      | B1+C1/2 | B2+C2/2 |
            | C3 | D3 | C4 | D4 |      | B3+C3/2 | B4+C4/2 |
            -------------      -------------
    """
    if not isinstance(image, Tensor) or image.ndim != 4:
        raise TypeError(
            f"Expected 'image' to be a 4D tensor, "
            f"but got {image.ndim}D {type(image).__name__},"
        )

    b, c, h, w  = image.shape
    device, dtype = image.device, image.dtype

    # Define kernels: filter_ad picks (top-left, bottom-right), filter_bc picks (top-right, bottom-left)
    # We use .repeat(c, 1, 1, 1) for channel-wise (depthwise) convolution
    kernel_ad = torch.tensor([[[[0.5, 0.0], [0.0, 0.5]]]], device=device, dtype=dtype).repeat(c, 1, 1, 1)
    kernel_bc = torch.tensor([[[[0.0, 0.5], [0.5, 0.0]]]], device=device, dtype=dtype).repeat(c, 1, 1, 1)

    # Stride=2 ensures non-overlapping 2x2 patches
    out_ad = F.conv2d(image, kernel_ad, stride=2, groups=c)
    out_bc = F.conv2d(image, kernel_bc, stride=2, groups=c)
    return out_ad, out_bc

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
