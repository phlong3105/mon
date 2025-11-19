#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements image I/O operations.

Common Tasks:
    - Load images from disk.
    - Save images to disk.
    - Batch I/O.
    - Metadata handling.
"""

__all__ = [
    "load",
    "read_shape",
    "save",
]

from typing import Union

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
    """Loads an image from a file path using OpenCV. Also add support for raw images.

    Args:
        path: Absolute path to the image file.
        flags: OpenCV flag to read image. One of: ``cv2.IMREAD_UNCHANGED``,
            ``cv2.IMREAD_GRAYSCALE``, ``cv2.IMREAD_COLOR_BGR``, ``cv2.IMREAD_COLOR``,
            ``cv2.IMREAD_ANYDEPTH``, ``cv2.IMREAD_ANYCOLOR``, ``cv2.IMREAD_COLOR_RGB``.
            Default: ``cv2.IMREAD_COLOR``.
    
    Returns:
        An RGB or grayscale image as a ``numpy.ndarray`` of shape :math:`(H, W, C)`
        in :math:`[0, 255]`.
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
        path: Absolute path to the image file.

    Returns:
        A tuple of :math:`(height, width, channels)`.

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


# ----- Writing -----
def save(image: Union[torch.Tensor, np.ndarray], path: Path):
    """Save an image to a file.

    Args:
        image: An RBG image as a
            ``torch.Tensor`` (i.e., of shape :math:`(B, C, H, W)` in :math:`[0.0, 1.0]`)
            or ``numpy.ndarray`` (i.e., of shape :math:`(H, W, C)` in :math:`[0, 255]`).
            Prioritize ``numpy.ndarray``.
        path: Absolute path to save the image file.

    Raises:
        TypeError: If ``image`` is not a ``torch.Tensor`` or ``numpy.ndarray``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(image, torch.Tensor):
        torchvision.utils.save_image(image, str(path))
    elif isinstance(image, np.ndarray):
        cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    else:
        raise TypeError(f"``image`` must be a torch.Tensor or numpy.ndarray, got {type(image)}.")
