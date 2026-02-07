#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Image I/O operations.

This module provides input and output operations for images.
"""

from __future__ import annotations

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


# ==============================================================================
# region DISCOVERY
# ==============================================================================


# endregion


# ==============================================================================
# region CONNECTION
# ==============================================================================


# endregion


# ==============================================================================
# region INPUT
# ==============================================================================

_PIL_MODE_TO_CHANNELS = {
    "1"    : 1,
    "L"    : 1,
    "P"    : 1,
    "RGB"  : 3,
    "RGBA" : 4,
    "CMYK" : 4,
    "YCbCr": 3,
    "LAB"  : 3,
    "HSV"  : 3,
    "I"    : 1,
    "F"    : 1,
}


def read(path: Path | str, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """Read an image from a file path.

    Read an image from a ``path`` using OpenCV. Also add support for raw images.

    Args:
        path: Absolute path to the image file.
        flags: OpenCV flag to read image. Defaults to cv2.IMREAD_COLOR.

    Returns:
        RGB or grayscale image, formatted as a numpy.ndarray of shape (H, W, C)
        and pixel values ranging from 0 to 255.

    Raises:
        RuntimeError: If OpenCV could not decode the image.
    """
    path = Path(path).normalize(exist=True)

    # Handle RAW Images
    if path.is_raw_image_file():  # Read raw image
        with rawpy.imread(str(path)) as raw:
            # use_camera_wb=True often provides a more natural look
            image = raw.postprocess(use_camera_wb=True, no_auto_bright=False, bright=1.0)

    # Handle Standard Images (OpenCV)
    else:
        # OpenCV reads BGR by default
        image = cv2.imread(str(path), flags)
        if image is None:
            raise RuntimeError(f"OpenCV could not decode image at: {path}")

        # Standardize dimensions: [H, W] -> [H, W, 1]
        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1)

        # Standardize color: BGR(A) -> RGB(A)
        if flags != cv2.IMREAD_GRAYSCALE:
            channels = image.shape[-1]
            if channels == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif channels == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

    return image


def read_shape(path: Path | str) -> tuple[int, int, int]:
    """Read the image's shape as (H, W, C) from a file path.

    Args:
        path: Absolute path to the image file.

    Raises:
        ValueError: If image mode is unsupported for non-RAW images.
    """
    path = Path(path).normalize(exist=True)

    # Handle RAW Images
    if path.is_raw_image_file():
        with rawpy.imread(str(path)) as raw:
            # Visible dimensions ignore the 'black' masked pixels at sensor edges
            h, w = raw.raw_image_visible.shape
            # Most RAW post-processing yields 3 channels (RGB)
            c    = 3

    # Handle Standard Images (using lazy-load PIL)
    else:
        with PIL.Image.open(str(path)) as img:
            w, h = img.size
            c    = _PIL_MODE_TO_CHANNELS.get(img.mode)
            if c is None:
                raise ValueError(f"Unsupported 'mode': {img.mode} for image at: {path}.")

    return h, w, c


def read_size(path: Path | str) -> tuple[int, int]:
    """Read the image's size as (H, W) from a file path.

    Args:
        path: Absolute path to the image file.
    """
    h, w, _ = read_shape(path=path)
    return h, w

# endregion


# ==============================================================================
# region OUTPUT
# ==============================================================================

def write(image: np.ndarray | torch.Tensor, path: Path | str):
    """Save an ``image`` to a ``path`` on disk.

    Args:
        image: RGB or grayscale image, formatted as a numpy.ndarray of shape
            (H, W, C) and pixel values ranging from 0 to 255; or as a
            torch.Tensor of shape (B, C, H, W) and pixel values ranging from
            0.0 to 1.0.
        path: Absolute path to save the ``image``. The parent directories will
            be created if they do not exist.

    Raises:
        TypeError: If ``image`` is not a torch.Tensor or numpy.ndarray.
    """
    path = Path(path).normalize(exist=False, mkdir=True)

    # Handle tensors (B, C, H, W)
    if isinstance(image, torch.Tensor):
        # torchvision handles the [0, 1] -> [0, 255] conversion internally
        # We ensure it's on CPU before saving
        torchvision.utils.save_image(image.cpu(), str(path))

    # Handle NumPy (H, W, C)
    elif isinstance(image, np.ndarray):
        # Ensure it's 8-bit for OpenCV
        if image.dtype != np.uint8:
            if image.max() <= 1.01:  # Check if it's normalized 0-1
                image = (image * 255).clip(0, 255)
            else:
                image = image.clip(0, 255)
            image = image.astype(np.uint8)

        # Standardize Color: RGB -> BGR (OpenCV default)
        if image.ndim == 3:
            if image.shape[-1] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            elif image.shape[-1] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)

        cv2.imwrite(str(path), image)
    else:
        raise TypeError(
            f"Expected 'image' to be a torch.Tensor or numpy.ndarray, "
            f"but got {type(image).__name__}."
        )

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
