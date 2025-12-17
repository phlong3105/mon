#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for image utilities.

This module implements utility functions for handling image data types,
including accessing image properties and validating image formats.
"""

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
    "shape",
]

import math
from typing import Any, Union

import numpy as np
import torch


# ----- Accessing -----
def center(image: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Retrieves the center of an image as (H/2, W/2).

    Args:
        image (torch.Tensor or numpy.ndarray): Image as a torch.Tensor
            (i.e., of shape (B, C, H, W) with pixel values in [0.0, 1.0]) or
            numpy.ndarray (i.e., of shape (H, W, C) with pixel values in
            [0, 255]).
    
    Returns:
        torch.Tensor or numpy.ndarray: Center of an image.
    """
    h, w    = imgsz(image)
    center_ = [h / 2, w / 2]
    return torch.tensor(center_) if isinstance(image, torch.Tensor) else np.array(center_)

    
def num_channels(image: torch.Tensor | np.ndarray) -> int:
    """Retrieves the number of channels in an image.

    Args:
        image (torch.Tensor or numpy.ndarray): Image as a torch.Tensor
            (i.e., of shape (B, C, H, W) with pixel values in [0.0, 1.0]) or
            numpy.ndarray (i.e., of shape (H, W, C) with pixel values in [0, 255]).
   
    Returns:
        int: Number of channels in the image. 
    """
    if image.ndim == 4:
        c = image.shape[1] if is_channel_first(image) else image.shape[3]
    elif image.ndim == 3:
        c = image.shape[0] if is_channel_first(image) else image.shape[2]
    elif image.ndim == 2:
        c = 1
    else:
        c = 0
    return c


def shape(image: torch.Tensor | np.ndarray) -> tuple[int]:
    """Retrieves height, width, and channels of an image.

    Args:
        image (torch.Tensor or numpy.ndarray): Image as a torch.Tensor
            (i.e., of shape (B, C, H, W) with pixel values in [0.0, 1.0]) or
            numpy.ndarray (i.e., of shape (H, W, C) with pixel values in [0, 255]).

    Returns:
        tuple[int]: Tuple of (H, W, C).
    """
    return (
        (image.shape[-2], image.shape[-1], image.shape[-3])
        if is_channel_first(image)
        else (image.shape[-3], image.shape[-2], image.shape[-1])
    )


def imgsz(image_or_size: Any, divisor: int = None) -> tuple[int, int]:
    """Retrieves the height and width of an image.

    Args:
        image_or_size (Any): Input image as a torch.Tensor or numpy.ndarray,
            or size as int, Sequence[int].
        divisor (int, optional): Divisor size for height and width.

    Returns:
        tuple[int, int]: Tuple of (H, W).

    Raises:
        TypeError: If ``image_or_size`` is not a valid type.
    """
    size = None
    if isinstance(image_or_size, list | tuple):
        if len(image_or_size) == 1:
            size = (image_or_size[0], image_or_size[0])
        elif len(image_or_size) == 2:
            size = image_or_size
        elif len(image_or_size) == 3:
            size = image_or_size[:2] if len(image_or_size) == 3 and image_or_size[0] >= image_or_size[2] else image_or_size[-2:]
    elif isinstance(image_or_size, (int, float)):
        size = (image_or_size, image_or_size)
    elif isinstance(image_or_size, torch.Tensor | np.ndarray):
        size = (
            (int(image_or_size.shape[-2]), int(image_or_size.shape[-1]))
            if is_channel_first(image_or_size)
            else (int(image_or_size.shape[-3]), int(image_or_size.shape[-2]))
        )
    else:
        raise TypeError(f"``input`` must be a torch.Tensor, numpy.ndarray, int, "
                        f"Sequence[int], str, or core.Path, got {type(image_or_size)}.")

    if divisor is not None:
        size = tuple(int(math.ceil(dim / divisor) * divisor) for dim in size)
    return size


# ----- Validation -----
def is_image(image: torch.Tensor | np.ndarray) -> bool:
    """Checks if an input is an image.

    Args:
        image (torch.Tensor or numpy.ndarray): Image as a torch.Tensor
            (i.e., of shape (B, C, H, W) with pixel values in [0.0, 1.0]) or
            numpy.ndarray (i.e., of shape (H, W, C) with pixel values in [0, 255]).
            
    Returns:
        bool: True if the input is an image, otherwise False.
    """
    return (
        isinstance(image, torch.Tensor | np.ndarray)
        and (is_color(image) or is_grayscale(image))
    )


def is_channel_first(image: torch.Tensor | np.ndarray) -> bool:
    """Checks if an image is in channel-first format.

    Args:
        image (torch.Tensor or numpy.ndarray): Image as a torch.Tensor
            (i.e., of shape (B, C, H, W) with pixel values in [0.0, 1.0]) or
            numpy.ndarray (i.e., of shape (H, W, C) with pixel values in [0, 255]).
            
    Returns:
        bool: True if image is in channel-first format, otherwise False.

    Raises:
        ValueError: If unable to determine channel format.

    Notes:
        Assumes the smallest dimension is the channel dimension.
    """
    # Determine tensor type and get shape
    if isinstance(image, torch.Tensor):
        shape_ = image.size()
    elif isinstance(image, np.ndarray):
        shape_ = image.shape
    else:
        raise TypeError(f"``image`` must be a numpy.ndarray or torch.Tensor, got {type(image)}.")
    
    # Check if tensor has at least 3 dimensions (batch, height/width, channels)
    if not 3 <= len(shape_) <= 4:
        raise ValueError(f"``image`` must have at least 3 dimensions, got {len(shape_)}.")
    
    # Extract dimensions
    if len(shape_) == 3:
        s0, s1, s2    = shape_
    else:
        _, s0, s1, s2 = shape_
    
    # Heuristic: Channels are typically smaller than spatial dimensions
    if (s0 < s1) and (s0 < s2):
        return True
    elif (s2 < s0) and (s2 < s1):
        return False
    else:
        raise ValueError(f"Cannot determine channel format for shape [{shape_}].")


def is_channel_last(image: torch.Tensor | np.ndarray) -> bool:
    """Checks if an image is in channel-last format.

    Args:
        image (torch.Tensor or numpy.ndarray): Image as a torch.Tensor
            (i.e., of shape (B, C, H, W) with pixel values in [0.0, 1.0]) or
            numpy.ndarray (i.e., of shape (H, W, C) with pixel values in [0, 255]).
            
    Returns:
        bool: True if image is in channel-last format, otherwise False.
    """
    return not is_channel_first(image)


def is_color(image: torch.Tensor | np.ndarray) -> bool:
    """Checks if an image is a color image.

    Args:
        image (torch.Tensor or numpy.ndarray): Image as a torch.Tensor
            (i.e., of shape (B, C, H, W) with pixel values in [0.0, 1.0]) or
            numpy.ndarray (i.e., of shape (H, W, C) with pixel values in [0, 255]).

    Returns:
        bool: True if the image has 3 or 4 channels, False otherwise.

    Notes:
        Assumes a color image has 3 or 4 channels (e.g., RGB or RGBA).
    """
    return num_channels(image) in [3, 4]


def is_grayscale(image: torch.Tensor | np.ndarray) -> bool:
    """Checks if an image is grayscale.

    Args:
        image (torch.Tensor or numpy.ndarray): Image as a torch.Tensor
            (i.e., of shape (B, C, H, W) with pixel values in [0.0, 1.0]) or
            numpy.ndarray (i.e., of shape (H, W, C) with pixel values in [0, 255]).
   
    Returns:
        bool: True if the image has 1 channel or is 2D, False otherwise.
    """
    return num_channels(image) == 1 or len(image.shape) == 2


def is_normalized(image: torch.Tensor | np.ndarray) -> bool:
    """Checks if an image is normalized to range [-1.0, 1.0] or [0.0, 1.0].

    Args:
        image (torch.Tensor or numpy.ndarray): Image as a torch.Tensor
            (i.e., of shape (B, C, H, W) with pixel values in [0.0, 1.0]) or
            numpy.ndarray (i.e., of shape (H, W, C) with pixel values in [0, 255]).
    
    Returns:
        bool: True if the image is normalized, False otherwise.
    
    Raises:
        TypeError: If ``image`` is neither a torch.Tensor nor a numpy.ndarray.
    """
    if isinstance(image, torch.Tensor):
        return abs(torch.max(image)) <= 1.0
    elif isinstance(image, np.ndarray):
        return abs(np.amax(image)) <= 1.0
    else:
        raise TypeError(f"``image`` must be a torch.Tensor or numpy.ndarray, got {type(image)}.")
