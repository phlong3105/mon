#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Image Processing Operations.

This module provides operations for processing images.
"""

from __future__ import annotations

__all__ = [
    "is_image",
    "parse_image_shape",
    "parse_imgsz",
    "to_image_array",
    "to_image_tensor",
]

from typing import Any

import torch
from numpy import ndarray
from torch import Tensor

from mon.core import Int3, Size, TensorOrArray


# ==============================================================================
# region VALIDATION
# ==============================================================================

def is_image(data: TensorOrArray, strict_channels: bool = True) -> bool:
    """Checks if the input is a structurally valid image tensor or array.

    Args:
        data (TensorOrArray): Image, formatted as a tensor of shape (B, C, H, W)
            and values ranging from 0.0 to 1.0; or as an array of shape (H, W, C)
            and values ranging from 0 to 255.
        strict_channels (bool, optional): If True, enforces that the channel
            dimension must be 1, 3, or 4. Set to False if you are using
            hyperspectral/medical images. Defaults to True.

    Returns:
        bool: True if it is a valid image representation, False otherwise.
    """
    # 1. Check valid data types
    if not isinstance(data, (ndarray, Tensor)):
        return False

    ndim = data.ndim
    shape = data.shape

    # 2. Images must be:
    #   - 2D (H, W),
    #   - 3D (C, H, W) / (H, W, C), or
    #   - 4D (B, C, H, W) / (B, H, W, C)
    if ndim not in (2, 3, 4):
        return False

    # 3. 2D images (Grayscale) are always valid
    if ndim == 2:
        return True

    # 4. Check for valid channel dimensions (1=Gray, 3=RGB, 4=RGBA)
    if strict_channels:
        valid_channels = (1, 3, 4)

        if ndim == 3:
            # Check if either the first dim (PyTorch) or last dim (NumPy)
            # is a valid channel
            is_chw = shape[0] in valid_channels
            is_hwc = shape[-1] in valid_channels
            if not (is_chw or is_hwc):
                return False
        elif ndim == 4:
            # Check if either the second dim (PyTorch: B, C, H, W)
            # or last dim (NumPy: B, H, W, C) is a valid channel
            is_bchw = shape[1] in valid_channels
            is_bhwc = shape[-1] in valid_channels
            if not (is_bchw or is_bhwc):
                return False

    # 5. Ensure spatial dimensions are strictly greater than 0
    # (Catches edge cases where an empty array was initialized)
    if 0 in shape:
        return False

    return True

# endregion


# ==============================================================================
# region RETRIEVAL
# ==============================================================================

# --- Accessing ---

def parse_image_shape(image: TensorOrArray) -> Int3:
    """Extract the shape of an image as a tuple of (H, W, C).

    Args:
        image (TensorOrArray): Image, formatted as a tensor of shape (B, C, H, W)
            and values ranging from 0.0 to 1.0; or as an array of shape (H, W, C)
            and values ranging from 0 to 255.

    Returns:
        Int3: Image shape as (H, W, C).

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

    raise ValueError(f"could not get shape from {type(image).__name__}.")


def parse_imgsz(value: Any, divisor: int | None = None) -> Size:
    """Extract the size of an image as a tuple of (H, W).

    Args:
        value (Any): Size-like (i.e., scalar or sequence) or an image.
        divisor (int | None, optional): Divisor size for height and width.
            Defaults to None.

    Returns:
        Size: Image size as (H, W).
    """
    return Size.from_value(value=value, divisor=divisor)


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
        TypeError: If ``image`` is not a 3D or 4D tensor.

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
            f"expected image to be a 4D tensor, "
            f"got {image.ndim}D {type(image).__name__}."
        )

    # If 4D, select the first image in the batch
    # We avoid squeeze() to prevent accidentally removing C=1
    image = image[0].detach().cpu()  # Move to CPU
    # (C, H, W) -> (H, W, C)
    image = image.permute(1, 2, 0).clamp(0.0, 1.0).mul(255).round().to(torch.uint8)
    return image.numpy()


def to_image_tensor(image: ndarray, normalize: bool = False) -> Tensor:
    """Convert an image from array to tensor.

    Args:
        image: Image array of shape (H, W, C) and values ranging from 0 to 255.
        normalize (bool, optional): If True, scales pixel values to range [0.0, 1.0].
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
            f"expected image to be a 4D tensor, "
            f"got {image.ndim}D {type(image).__name__}."
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


# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
