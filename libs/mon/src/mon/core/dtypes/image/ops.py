#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Image atomic operations.

This module provides atomic operations for images.
"""

from __future__ import annotations

__all__ = [
    "center",
    "imgsz",
    "is_channel_first",
    "is_channel_last",
    "is_color",
    "is_grayscale",
    "is_image",
    "is_normalized",
    "num_channels",
    "pad_square",
    "pair_downsample",
    "shape",
    "split",
    "to_array",
    "to_tensor",
]

import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


# ==============================================================================
# region CREATION
# ==============================================================================


# endregion


# ==============================================================================
# region VALIDATION
# ==============================================================================

def is_image(image: np.ndarray | torch.Tensor) -> bool:
    """Check if the input is an image.

    Args:
        image: An RGB or grayscale image, formatted as a numpy.ndarray of shape
            (H, W, C) and pixel values ranging from 0 to 255; or as a
            torch.Tensor of shape (B, C, H, W) and pixel values ranging
            from 0.0 to 1.0.

    Returns:
        True if the input is an image, otherwise False.
    """
    return (
        isinstance(image, (np.ndarray, torch.Tensor))
        and (is_color(image) or is_grayscale(image))
    )


def is_channel_first(image: np.ndarray | torch.Tensor) -> bool:
    """Check if an image is in channel-first format.

    Args:
        image: An RGB or grayscale image, formatted as a numpy.ndarray with
            pixel values ranging from 0 to 255; or as a torch.Tensor with pixel
            values ranging from 0.0 to 1.0.

    Returns:
        True if the ``image`` is in channel-first format, otherwise False.

    Raises:
        ValueError: If ``image`` does not have 3 or 4 dimensions.
    """
    shape_ = image.shape if isinstance(image, np.ndarray) else image.size()

    # Handle Batch vs No-Batch
    if len(shape_) == 4:
        # (B, C, H, W) vs (B, H, W, C)
        c_candidate_first = shape_[1]
        c_candidate_last  = shape_[3]
    elif len(shape_) == 3:
        # (C, H, W) vs (H, W, C)
        c_candidate_first = shape_[0]
        c_candidate_last  = shape_[2]
    else:
        raise ValueError(f"Expected 'image' to have 3 or 4 dimensions, "
                         f"but got {len(shape_)}.")

    # Standard color channel counts
    common_channels = {1, 2, 3, 4}  # 2 for optical flow, 1,3,4 for images

    if c_candidate_first in common_channels and c_candidate_last not in common_channels:
        return True
    if c_candidate_last in common_channels and c_candidate_first not in common_channels:
        return False

    # Size-based heuristic if both or neither match common counts
    return c_candidate_first < shape_[-2] and c_candidate_first < shape_[-1]


def is_channel_last(image: np.ndarray | torch.Tensor) -> bool:
    """Check if an image is in channel-last format.

    Args:
        image: An RGB or grayscale image, formatted as a numpy.ndarray with
            pixel values ranging from 0 to 255; or as a torch.Tensor with pixel
            values ranging from 0.0 to 1.0.

    Returns:
        True if the ``image`` is in channel-last format, otherwise False.
    """
    return not is_channel_first(image)


def is_color(image: np.ndarray | torch.Tensor) -> bool:
    """Check if an image is a color image.

    Args:
        image: An RGB or grayscale image, formatted as a numpy.ndarray of shape
            (H, W, C) and pixel values ranging from 0 to 255; or as a
            torch.Tensor of shape (B, C, H, W) and pixel values ranging from
            0.0 to 1.0.

    Returns:
        True if the ``image`` has 3 or 4 channels, False otherwise.

    Notes:
        Assumes a color image has 3 or 4 channels (e.g., RGB or RGBA).
    """
    return num_channels(image) in [3, 4]


def is_grayscale(image: np.ndarray | torch.Tensor) -> bool:
    """Check if an image is a grayscale image.

    Args:
        image: An RGB or grayscale image, formatted as a numpy.ndarray of shape
            (H, W, C) and pixel values ranging from 0 to 255; or as a
            torch.Tensor of shape (B, C, H, W) and pixel values ranging from
            0.0 to 1.0.

    Returns:
        True if the ``image`` has 1 channel or is 2D, False otherwise.
    """
    return num_channels(image) == 1 or len(image.shape) == 2


def is_normalized(image: np.ndarray | torch.Tensor) -> bool:
    """Check if an image is normalized to the range [-1, 1] or [0, 1].

    Args:
        image: An RGB or grayscale image, formatted as a numpy.ndarray of shape
            (H, W, C) and pixel values ranging from 0 to 255; or as a
            torch.Tensor of shape (B, C, H, W) and pixel values ranging from
            0.0 to 1.0.

    Returns:
        True if the ``image`` is normalized, False otherwise.
    """
    # Check dtype first (fastest)
    if isinstance(image, np.ndarray) and image.dtype == np.uint8:
        return False

    # Check values
    if isinstance(image, torch.Tensor):
        return image.max().item() <= 1.01  # Tolerance for float precision
    return np.max(image) <= 1.01

# endregion


# ==============================================================================
# region RETRIEVAL
# ==============================================================================

# --- Accessing ---

def shape(image: np.ndarray | torch.Tensor) -> tuple[int, int, int]:
    """Extract the shape of an image as (H, W, C).

    Args:
        image: An RGB or grayscale image, formatted as a numpy.ndarray of shape
            (H, W, C) and pixel values ranging from 0 to 255; or as a
            torch.Tensor of shape (B, C, H, W) and pixel values ranging from
            0.0 to 1.0.

    Returns:
        The shape of the image as (H, W, C).
    """
    # Handle 2D Grayscale case (H, W) -> (H, W, 1)
    if image.ndim == 2:
        return image.shape[0], image.shape[1], 1

    # Standard 3D/4D case
    return (
        (image.shape[-2], image.shape[-1], image.shape[-3])
        if is_channel_first(image)
        else (image.shape[-3], image.shape[-2], image.shape[-1])
    )


def imgsz(image_or_size: Any, divisor: int = None) -> tuple[int, int]:
    """Extract the size of an image as (H, W).

    Args:
        image_or_size: Image as a torch.Tensor or numpy.ndarray, or size as int,
            Sequence[int].
        divisor: Divisor size for height and width.

    Returns:
        The size of the image as (H, W).

    Raises:
        TypeError: If ``image_or_size`` is not a supported type.
    """
    size = None

    # Handle Tensors/Arrays
    if isinstance(image_or_size, (np.ndarray, torch.Tensor)):
        if is_channel_first(image_or_size):
            # Supports [C, H, W] or [B, C, H, W]
            size = (int(image_or_size.shape[-2]), int(image_or_size.shape[-1]))
        else:
            # Supports [H, W, C] or [B, H, W, C]
            size = (int(image_or_size.shape[-3]), int(image_or_size.shape[-2]))

    # Handle Numeric inputs
    elif isinstance(image_or_size, (int, float)):
        size = (int(image_or_size), int(image_or_size))

    # Handle Sequences
    elif isinstance(image_or_size, (list, tuple)):
        if len(image_or_size) >= 2:
            # Take the first two elements assuming they represent (H, W)
            size = (image_or_size[0], image_or_size[1])
        elif len(image_or_size) == 1:
            size = (image_or_size[0], image_or_size[0])

    if size is None:
        raise TypeError(f"Could not parse size from {type(image_or_size)}")

    # Apply Divisor (Rounding up to the nearest multiple)
    if divisor:
        h, w = size
        h    = int(math.ceil(h / divisor) * divisor)
        w    = int(math.ceil(w / divisor) * divisor)
        size = (h, w)

    return size


def num_channels(image: np.ndarray | torch.Tensor) -> int:
    """Extract the number of channels in an image.

    Args:
        image: An RGB or grayscale image, formatted as a numpy.ndarray of shape
            (H, W, C) and pixel values ranging from 0 to 255; or as a
            torch.Tensor of shape (B, C, H, W) and pixel values ranging from
            0.0 to 1.0.

    Returns:
        The number of channels in the image.
    """
    # 2D Grayscale case: Always 1 channel
    if image.ndim == 2:
        return 1

    # Batch (4D) or Single (3D) case
    if is_channel_first(image):
        # (B, C, H, W) or (C, H, W): Channel is at index 1 or 0 respectively.
        # Conveniently, it's always the dimension before H and W.
        return image.shape[-3]
    else:
        # (B, H, W, C) or (H, W, C): Channel is always last.
        return image.shape[-1]


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

def center(image: np.ndarray | torch.Tensor, integer: bool = True) -> np.ndarray | torch.Tensor:
    """Extract the center coordinates of an image.

    Args:
        image: An RGB or grayscale image, formatted as a numpy.ndarray of shape
            (H, W, C) and pixel values ranging from 0 to 255; or as a
            torch.Tensor of shape (B, C, H, W) and pixel values ranging from
            0.0 to 1.0.
        integer: If True, rounds the center coordinates to the nearest integer.
            Defaults to True.
    """
    h, w = imgsz(image)

    # Calculate center
    y_c, x_c = (h / 2, w / 2) if not integer else (h // 2, w // 2)
    center_  = [y_c, x_c]

    return torch.tensor(center_) if isinstance(image, torch.Tensor) else np.array(center_)

# endregion


# ==============================================================================
# region TRANSFORMATION
# ==============================================================================

# --- Casting ---

def to_array(image: torch.Tensor) -> np.ndarray:
    """Convert an image from torch.Tensor to numpy.ndarray.

    Args:
        image: An RGB or grayscale image, formatted as a torch.Tensor of shape
            (B, C, H, W) and pixel values ranging from 0.0 to 1.0.

    Returns:
        An RGB or grayscale image, formatted as a numpy.ndarray of shape
            (H, W, C) and pixel values ranging from 0 to 255.

    Raises:
        TypeError: If ``image`` is not a 4D torch.Tensor.

    Notes:
        image = (tensor.squeeze().detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype("uint8")
    """
    if not torch.is_tensor(image) or image.ndim != 4:
        raise TypeError(f"Expected 'image' to be a 4D torch.Tensor, "
                        f"but got {image.ndim}D.")

    # Select the first image in batch if B > 1, then move to CPU
    # We avoid squeeze() to prevent accidentally removing C=1
    img = image[0].detach().cpu()

    # Perform operations on the device to minimize transfer overhead
    # Select the first image in batch if B > 1
    image = image[0].detach()
    # [C, H, W] -> [H, W, C]
    image = image.permute(1, 2, 0).clamp(0, 1).mul(255).round().to(torch.uint8)
    return image.cpu().numpy()


def to_tensor(image: np.ndarray, normalize: bool = False) -> torch.Tensor:
    """Convert an image from numpy.ndarray to torch.Tensor.

    Args:
        image: An RGB or grayscale image, formatted as a numpy.ndarray of shape
            (H, W, C) and pixel values ranging from 0 to 255.
        normalize: If True, scales pixel values to range [0.0, 1.0]. Defaults to False.

    Returns:
        An RGB or grayscale image, formatted as a torch.Tensor of shape
            (B, C, H, W) and pixel values ranging from 0.0 to 1.0 if ``normalize``
            is True, else ranging from 0 to 255.

    Raises:
        TypeError: If ``image`` is not a 3D numpy.array.

    Notes:
        image = torch.from_numpy(image).permute(2, 0, 1).contiguous().float().div(255.0).unsqueeze(0).to(device)
    """
    if not isinstance(image, np.ndarray) or image.ndim != 3:
        raise TypeError(f"Expected 'image' to be a 3D numpy.ndarray, "
                        f"but got {image.ndim}D.")

    # Convert to tensor and permute: [H, W, C] -> [C, H, W]
    tensor = torch.from_numpy(image).permute(2, 0, 1).float()

    if normalize:
        tensor = tensor.div(255.0)

    # Add batch dimension: [1, C, H, W]
    return tensor.unsqueeze(0).contiguous()


# --- Encoding ---


# --- Standardization ---


# --- Structural ---

def split(image: np.ndarray, n: int = 2) -> list[np.ndarray]:
    """Split an image into ``n`` equal parts.

    Args:
        image: An RGB or grayscale image, formatted as a numpy.ndarray of shape
            (H, W, C) and pixel values ranging from 0 to 255.
        n: Number of parts to split the image into. Default to 2.

    Returns:
        List of ``n`` sub-images as numpy.ndarray of shape approximately (H/n, W/n, C).

    Raises:
        ValueError: If ``image`` is not a 3D numpy.ndarray.
    """
    if image.ndim != 3:
        raise ValueError(f"Expected 'image' to be a 3D numpy.ndarray, but got {image.ndim}D.")

    h, w, c = image.shape

    # Find optimal Grid (Rows, Cols)
    # We look for factors of n that best match the image aspect ratio
    best_ratio_diff = float("inf")
    rows, cols = 1, n
    img_aspect = h / w

    for r in range(1, n + 1):
        if n % r == 0:
            c_grid = n // r
            grid_aspect = r / c_grid
            # We want the sub-image aspect (h/r) / (w/c) to be near 1 (square)
            # Or match the orientation preference
            ratio_diff = abs(grid_aspect - img_aspect)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                rows, cols = r, c_grid

    # Extract Tiles
    sub_images = []
    # Use np.array_split to handle uneven sizes automatically
    # This ensures h % rows pixels are distributed across tiles
    row_splits = np.array_split(np.arange(h), rows)
    col_splits = np.array_split(np.arange(w), cols)

    for r_indices in row_splits:
        for c_indices in col_splits:
            tile = image[r_indices[0]:r_indices[-1]+1,
                         c_indices[0]:c_indices[-1]+1]
            sub_images.append(tile)

    return sub_images


# --- Statistical ---


# --- Geometric ---

def pad_square(
    image    : np.ndarray,
    pad_value: int = 0,
    mode     : str = "constant"
) -> np.ndarray:
    """Pad an image to make it a square.

    Args:
        image: An RGB or grayscale image, formatted as a numpy.ndarray of shape
            (H, W, C) and pixel values ranging from 0 to 255.
        pad_value: Padding value. Default to 0.
        mode: Padding mode. Defaults to "constant".

    Returns:
        Padded square image of shape (S, S, C), where S is the maximum of (H, W).

    Raises:
        ValueError: If ``image`` is not a 2D or 3D numpy.ndarray.
    """
    if image.ndim not in [2, 3]:
        raise ValueError(f"Expected 'image' to be a 2D or 3D numpy.ndarray, "
                         f"but got {image.ndim}D.")

    h, w = image.shape[:2]
    size = max(h, w)

    # Calculate padding for top, bottom, left, right
    pad_h = size - h
    pad_w = size - w

    top, bottom = pad_h // 2, pad_h - (pad_h // 2)
    left, right = pad_w // 2, pad_w - (pad_w // 2)

    # Construction of padding width tuple
    # For (H, W, C): ((top, bottom), (left, right), (0, 0))
    pad_width = [(top, bottom), (left, right)]
    if image.ndim == 3:
        pad_width.append((0, 0))  # Don't pad the channel dimension

    if mode == "constant":
        return np.pad(image, pad_width, mode=mode, constant_values=pad_value)
    else:
        return np.pad(image, pad_width, mode=mode)


def pair_downsample(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Downsample an image tensor into a pair to half resolution.

    References:
        - Code: https://colab.research.google.com/drive/1i82nyizTdszyHkaHBuKPbWnTzao8HF9b?usp=sharing

    Args:
        image: An RGB or grayscale image, formatted as a torch.Tensor of shape
            (B, C, H, W) and pixel values ranging from 0.0 to 1.0.

    Returns:
        Tuple containing two downsampled images.

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
    if image.ndim != 4:
        raise TypeError(f"Expected 'image' to be a 4D torch.Tensor, but got {image.ndim}D.")

    b, c, h, w    = image.shape
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
# region DESTRUCTION
# ==============================================================================


# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
