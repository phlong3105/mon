#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for image I/O operations.

This module provides functions to load and save images using OpenCV, PIL, and
rawpy. It supports various image formats, including raw images, and provides
utilities to read image shape and size.
"""

__all__ = [
    "load",
    "read_shape",
    "read_size",
    "save",
]

import cv2
import numpy as np
import PIL.Image
import rawpy
import torch
import torchvision

from mon.core.pathlib import Path
from .utils import is_color


# ----- Reading -----
def load(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """Loads an image from a file path using OpenCV. Also add support for raw
    images.

    Args:
        path (Path): Absolute path to the image file.
        flags (int): OpenCV flag to read image. Defaults to cv2.IMREAD_COLOR.
    
    Returns:
        np.ndarray: An RGB image as a numpy.ndarray of shape (H, W, C) with pixel
            values in the range [0, 255].
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
    """Reads an image shape from a file path using PIL or rawpy.

    Args:
        path (Path): Absolute path to the image file.
        
    Returns:
        tuple[int, int, int]: A tuple of (H, W, C).
        
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
            mode = image.mode
            c = {"RGB": 3, "RGBA": 4, "L": 1}.get(mode, None)
            if c is None:
                raise ValueError(f"Unsupported image mode {mode}.")
    return h, w, c


def read_size(path: Path) -> tuple[int, int]:
    """Reads an image size from a file path using PIL or rawpy.

    Args:
        path (Path): Absolute path to the image file.
    
    Returns:
        tuple[int, int]: A tuple of (H, W).
    """
    return read_shape(path=path)[:2]


# ----- Writing -----
def save(image: torch.Tensor | np.ndarray, path: Path):
    """Saves an image to a file.

    Args:
        image (torch.Tensor or numpy.ndarray): An RGB image as a torch.Tensor or
            numpy.ndarray of shape (C, H, W) or (H, W, C) with pixel values in
            the range [0, 1] for torch.Tensor or [0, 255] for numpy.ndarray.
        path (Path): Absolute path to save the image. The parent directories
            will be created if they do not exist.

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
