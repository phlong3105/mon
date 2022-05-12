# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""Conversion between single-channel integer value to 3-channels color image.
Mostly used for semantic segmentation.
"""

from __future__ import annotations

import numpy as np
import torch
from multipledispatch import dispatch
from torch import Tensor

from one.core import get_num_channels
from one.core import TensorOrArray
from one.core import to_channel_first

__all__ = [
	"integer_to_color",
	"is_color_image",
]


# MARK: - Functional


def _integer_to_color(image: np.ndarray, colors: list) -> np.ndarray:
    """Convert the integer-encoded image to color image. Fill an image with
    labels' colors.

    Args:
        image (np.ndarray):
            An image in either one-hot or integer.
        colors (list):
            List of all colors.

    Returns:
        color (np.ndarray):
            Colored image.
    """
    if len(colors) <= 0:
        raise ValueError(f"No colors are provided.")
    
    # NOTE: Convert to channel-first
    image = to_channel_first(image)
    
    # NOTE: Squeeze dims to 2
    if image.ndim == 3:
        image = np.squeeze(image)
    
    # NOTE: Draw color
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    for l in range(0, len(colors)):
        idx = image == l
        r[idx] = colors[l][0]
        g[idx] = colors[l][1]
        b[idx] = colors[l][2]
    rgb = np.stack([r, g, b], axis=0)
    return rgb


@dispatch(Tensor, list)
def integer_to_color(image: Tensor, colors: list) -> Tensor:
    mask_np = image.numpy()
    mask_np = integer_to_color(mask_np, colors)
    color   = torch.from_numpy(mask_np)
    return color


@dispatch(np.ndarray, list)
def integer_to_color(image: np.ndarray, colors: list) -> np.ndarray:
    # [C, H, W]
    if image.ndim == 3:
        return _integer_to_color(image, colors)
    
    # [B, C, H, W]
    if image.ndim == 4:
        colors = [_integer_to_color(i, colors) for i in image]
        colors = np.stack(colors).astype(np.uint8)
        return colors
    
    raise ValueError(f"`image.ndim` must be 3 or 4. But got: {image.ndim}.")


def is_color_image(image: TensorOrArray) -> bool:
    """Check if the given image is color encoded."""
    if get_num_channels(image) in [3, 4]:
        return True
    return False
