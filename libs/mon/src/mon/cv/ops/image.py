#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Image Operation.

This module provides atomic operations for images.
"""

from __future__ import annotations

__all__ = [
    "pair_downsample",
    "parse_image_shape",
    "parse_imgsz",
    "to_image_array",
    "to_image_tensor",
]

import math
from typing import Any

import torch
from numpy import ndarray
from torch import Tensor
from torch.nn import functional as F

from mon.core import int_3_t, TensorOrArray


# ==============================================================================
# region VALIDATION
# ==============================================================================


# endregion


# ==============================================================================
# region RETRIEVAL
# ==============================================================================

# --- Accessing ---

def parse_image_shape(image: TensorOrArray) -> int_3_t:
    """Extract the shape of an image as a tuple of (H, W, C).

    Args:
        image (TensorOrArray): Image, formatted as a tensor of shape (B, C, H, W)
            and values ranging from 0.0 to 1.0; or as an array of shape (H, W, C)
            and values ranging from 0 to 255.

    Raises:
        ValueError: If ``image`` is not a supported shape.
    """
    # Handle 2D grayscale case (H, W) -> (H, W, 1)
    if image.ndim == 2:
        return image.shape[0], image.shape[1], 1

    # Standard 3D/4D case
    if image.ndim in [3, 4]:
        # Assume channel-first for Tensor, channel-last for NumPy
        if isinstance(image, Tensor):
            return image.shape[-2], image.shape[-1], image.shape[-3]
        elif isinstance(image, ndarray):
            return image.shape[-3], image.shape[-2], image.shape[-1]

    raise ValueError(f"Could not get shape from {type(image)}.")


def parse_imgsz(value: Any, divisor: int = None) -> tuple[int, int]:
    """Extract the size of an image as a tuple of (H, W).

    Args:
        value (Any): Size-like (i.e., scalar or sequence) or an image.
        divisor (int, optional): Divisor size for height and width.
            Defaults to None.

    Raises:
        TypeError: If ``image_or_size`` is not a supported type.
    """
    size = None

    # Handle Tensors/Arrays
    if isinstance(value, Tensor):
        size = (int(value.shape[-2]), int(value.shape[-1]))
    elif isinstance(value, ndarray):
        size = (int(value.shape[-3]), int(value.shape[-2]))

    # Handle scalars
    elif isinstance(value, (int, float)):
        size = (int(value), int(value))

    # Handle Sequences
    elif isinstance(value, (list, tuple)):
        if len(value) >= 2:
            # Take the first two elements assuming they represent (H, W)
            size = (value[0], value[1])
        elif len(value) == 1:
            size = (value[0], value[0])

    if size is None:
        raise TypeError(f"Could not get size from {type(value)}.")

    # Apply Divisor (Rounding up to the nearest multiple)
    if divisor:
        h, w = size
        h = int(math.ceil(h / divisor) * divisor)
        w = int(math.ceil(w / divisor) * divisor)
        size = (h, w)

    return size


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

def to_image_array(image: Tensor) -> ndarray:
    """Convert an image from tensor to an array.

    Args:
        image (Tensor): Image tensor of shape (B, C, H, W) and values ranging
            from 0.0 to 1.0.

    Returns:
        ndarray: Image array of shape (H, W, C) and values ranging from 0 to 255.

    Raises:
        TypeError: If ``image`` is not a 4D tensor.

    Notes:
        image = (tensor.squeeze().detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype("uint8")
    """
    if (
        not isinstance(image, Tensor)
        or image.ndim != 4
        or image.shape[0] != 1
        or image.shape[1] not in [1, 3, 4]
    ):
        raise TypeError(
            f"Expected 'image' to be a 4D tensor, "
            f"but got {image.ndim}D {type(image).__name__},"
        )

    # Select the first image in the batch if B > 1, then move to CPU
    # We avoid squeeze() to prevent accidentally removing C=1
    image = image[0].detach().cpu()
    # (C, H, W) -> (H, W, C)
    image = image.permute(1, 2, 0).clamp(0, 1).mul(255).round().to(torch.uint8)
    return image.numpy()


def to_image_tensor(image: ndarray, normalize: bool = False) -> Tensor:
    """Convert an image from array to tensor.

    Args:
        image: Image array of shape (H, W, C) and values ranging from 0 to 255.
        normalize (bool): If True, scales pixel values to range [0.0, 1.0].
            Defaults to False.

    Returns:
        Tensor: Image tensor of shape (B, C, H, W) and values ranging from
            0.0 to 1.0.

    Raises:
        TypeError: If ``image`` is not a 3D array.

    Notes:
        image = torch.from_numpy(image).permute(2, 0, 1).contiguous().float().div(255.0).unsqueeze(0).to(device)
    """
    if not isinstance(image, ndarray) or image.ndim != 3:
        raise TypeError(
            f"Expected 'image' to be a 4D tensor, "
            f"but got {image.ndim}D {type(image).__name__},"
        )

    # Convert to tensor and permute: [H, W, C] -> [C, H, W]
    tensor = torch.from_numpy(image).permute(2, 0, 1).float()

    # Normalize pixel values
    if normalize:
        tensor = tensor.div(255.0)

    # Add batch dimension: [1, C, H, W]
    return tensor.unsqueeze(0).contiguous()


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
