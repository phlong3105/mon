#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Image data I/O operations.

This module provides functions for input and output operations for image data.
"""

__all__ = [
    "read",
    "read_shape",
    "read_size",
    "write",
]

import cv2
import numpy as np
import PIL.Image
import rawpy
import torch
import torchvision

from mon.core.pathlib import Path
from .meta import is_color


# ==============================================================================
# RESOURCE RESOLVERS (Path/URL Handling)
# ==============================================================================

# --- Path Handling (Resolving URIs, Local Paths) ---


# --- Backend Selection (Selecting PIL vs. OpenCV vs. TurboJPEG) ---


# ==============================================================================
# HYDRATION & DESERIALIZATION (Read/Load)
# ==============================================================================

# --- Deserialize (Bytes to Object) ---


# --- Loaders (Standard Disk-to-RAM logic) ---
def read(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """Read an image from a file path using OpenCV. Also add support for raw
    images.
    
    Notes:
        If later, this function also includes normalization, tensor, etc., then
        it should be renamed to ``load``.
    
    Args:
        path: Absolute path to the image file.
        flags: OpenCV flag to read image. Defaults to cv2.IMREAD_COLOR.
    
    Returns:
        An RGB image of shape (H, W, C) with pixel values in the range [0, 255].
    """
    path = Path(path)
    if path.is_raw_image_file():  # Read raw image
        image = rawpy.imread(str(path))
        image = image.postprocess()
    else:  # Read other types of image
        image = cv2.imread(str(path), flags)  # BGR
        if image.ndim == 2:  # [H, W] -> [H, W, 1] for grayscale
            image = np.expand_dims(image, axis=-1)
        if is_color(image):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Safer
    
    return image


def read_shape(path: Path) -> tuple[int, int, int]:
    """Read the image's shape from a file path.

    Args:
        path: Absolute path to the image file.
        
    Returns:
        An image's shape as (H, W, C).
        
    Raises:
        ValueError: If image mode is unsupported for non-RAW images.
    """
    path = Path(path)
    if path.is_raw_image_file():
        image = rawpy.imread(str(path)).raw_image_visible
        h, w  = image.shape
        c     = 3
    else:
        with PIL.Image.open(str(path)) as image:
            w, h = image.size
            c    = {"RGB": 3, "RGBA": 4, "L": 1}.get(image.mode, None)
            if c is None:
                raise ValueError(f"Unsupported image mode {image.mode}.")
    return h, w, c


def read_size(path: Path) -> tuple[int, int]:
    """Read the image's size from a file path.

    Args:
        path: Absolute path to the image file.
    
    Returns:
         An image's size as (H, W).
    """
    shape = read_shape(path=path)
    return shape[0], shape[1]


# ==============================================================================
# PERSISTENCE & EXPORT (Write/Commit)
# ==============================================================================

# --- Serialize (Object to Bytes) ---


# --- Commit (Saving to Disk/Cloud) ---
def write(image: torch.Tensor | np.ndarray, path: Path):
    """Save an image to disk.

    Args:
        image: An RGB image as a torch.Tensor of shape (B, C, H, W) with pixel
            values in the range [0, 1] or a numpy.ndarray of shape (H, W, C)
            with pixel values in the range [0, 255].
        path: Absolute path to save the image. The parent directories will be
            created if they do not exist.

    Raises:
        TypeError: If ``image`` is not a torch.Tensor or numpy.ndarray.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(image, torch.Tensor):
        torchvision.utils.save_image(image, str(path))
    elif isinstance(image, np.ndarray):
        cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    else:
        raise TypeError(f"``image`` must be a torch.Tensor or numpy.ndarray, got {type(image)}.")
