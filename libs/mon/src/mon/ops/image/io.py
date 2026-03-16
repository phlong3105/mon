#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Image I/O Operations.

This module provides input and output operations for images.
"""

from __future__ import annotations

__all__ = [
    "ImageLoader",
    "MaskLoader",
    "read_image",
    "read_image_shape",
    "read_imgsz",
    "write_image",
]

from typing import override

import cv2
import numpy as np
import PIL.Image
import rawpy
import torchvision
from numpy import ndarray
from torch import Tensor

from mon.core import (
    Image,
    Loader,
    Metadata,
    Path,
    PathLike,
    singleton,
    Size,
    TensorOrArray,
)

# ==============================================================================
# region CONSTANTS
# ==============================================================================

_PIL_MODE_TO_CHANNELS = {
    "1": 1,
    "L": 1,
    "P": 1,
    "RGB": 3,
    "RGBA": 4,
    "CMYK": 4,
    "YCbCr": 3,
    "LAB": 3,
    "HSV": 3,
    "I": 1,
    "F": 1,
}

# endregion


# ==============================================================================
# region DISCOVERY
# ==============================================================================


# endregion


# ==============================================================================
# region INPUT
# ==============================================================================

def read_image(path: PathLike, flags: int = cv2.IMREAD_COLOR) -> ndarray:
    """Read an image from a path using OpenCV.

    Args:
        path (PathLike): Absolute path to the image file.
        flags (int): OpenCV flag to read the image. Defaults to cv2.IMREAD_COLOR.

    Returns:
        ndarray: Image array of shape (H, W, C) and pixel values ranging from
            0 to 255.

    Raises:
        RuntimeError: If OpenCV could not decode the image.
    """
    path = Path(path).normalize()

    if path.is_raw_image_file(exists=True):
        # Handle RAW Images
        with rawpy.imread(str(path)) as raw:
            # use_camera_wb=True often provides a more natural look
            image = raw.postprocess(
                use_camera_wb=True,
                no_auto_bright=False,
                bright=1.0
            )
    else:
        # Handle Standard Images (OpenCV)
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


def read_image_shape(path: PathLike) -> tuple[int, int, int]:
    """Read the image's shape as (H, W, C) from a file path.

    Args:
        path (PathLike): Absolute path to the image file.

    Returns:
        tuple[int, int, int]: Height, width, and number of channels of the image.

    Raises:
        ValueError: If image mode is unsupported for non-RAW images.
    """
    path = Path(path).normalize()

    if path.is_raw_image_file(exists=True):
        # Handle RAW Images
        with rawpy.imread(str(path)) as raw:
            # Visible dimensions ignore the 'black' masked pixels at sensor edges
            h, w = raw.raw_image_visible.shape
            # Most RAW post-processing yields 3 channels (RGB)
            c = 3
    else:
         # Handle Standard Images (using lazy-load PIL)
        with PIL.Image.open(str(path)) as img:
            w, h = img.size
            c  = _PIL_MODE_TO_CHANNELS.get(img.mode)
            if c is None:
                raise ValueError(
                    f"Unsupported 'mode': {img.mode} for image at: {path}."
                )

    return h, w, c


def read_imgsz(path: PathLike) -> Size:
    """Read the image's size as (H, W) from a file path.

    Args:
        path (PathLike): Absolute path to the image file.

    Returns:
        Size: Height and width of the image.
    """
    h, w, _ = read_image_shape(path=path)
    return Size(height=h, width=w)


@singleton
class ImageLoader(Loader):
    """Image data loader.

    Extend the ``DataLoader`` class to load images from disk. Make this class a
    singleton to ensure that only a single instance of the loader is created.
    """

    # --- Input ---
    @override
    def load(
        self,
        metadata: Metadata | None,
        flags: int = cv2.IMREAD_COLOR,
    ) -> Image | None:
        """Load image data from the given ``metadata``.

        Args:
            metadata (Metadata, optional): Metadata describing the image to be
                loaded. Defaults to None.
            flags (int): OpenCV flag to read the image. Defaults to cv2.IMREAD_COLOR.

        Returns:
            Image | None: An ``Image`` instance containing the loaded image,
                or None if loading failed.
        """
        # Validate inputs
        if metadata is None:
            return None

        image = read_image(path=metadata.path, flags=flags)
        return Image(image=image, path=metadata.path, base_dir=metadata.base_dir)


@singleton
class MaskLoader(Loader):
    """Mask data loader.

    Extend the ``DataLoader`` class to load masks (e.g., segmentation masks,
    depth maps, etc.) from disk. Make this class a singleton to ensure that only
    a single instance of the loader is created.
    """

    # --- Input ---
    @override
    def load(
        self,
        metadata: Metadata | None,
        flags: int = cv2.IMREAD_GRAYSCALE,
    ) -> Image | None:
        """Load image data from the given ``metadata``.

        Args:
            metadata (Metadata, optional): Metadata describing the image to be
                loaded. Defaults to None.
            flags (int): OpenCV flag to read the image. Defaults to cv2.IMREAD_GRAYSCALE.

        Returns:
            Image | None: An ``Image`` instance containing the loaded image,
                or None if loading failed.
        """
        # Validate inputs
        if metadata is None:
            return None

        image = read_image(path=metadata.path, flags=flags)
        return Image(image=image, path=metadata.path, base_dir=metadata.base_dir)

# endregion


# ==============================================================================
# region OUTPUT
# ==============================================================================

def write_image(image: TensorOrArray, path: PathLike):
    """Save an image to a file path on disk.

    Args:
        image (TensorOrArray): Image, formatted as a tensor of shape (B, C, H, W)
            and values ranging from 0.0 to 1.0; or as an array of shape (H, W, C)
            and values ranging from 0 to 255.
        path (PathLike): Absolute path to save the image.

    Raises:
        TypeError: If ``image`` is not a tensor or array.
    """
    # Normalize inputs
    path = Path(path).normalize()

    # Create parent directory
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(image, Tensor):
        # Handle tensor (B, C, H, W)
        # torchvision handles the [0, 1] -> [0, 255] conversion internally
        # We ensure it's on CPU before saving
        torchvision.utils.save_image(image.detach().cpu(), str(path))
    elif isinstance(image, ndarray):
        # Handle array (H, W, C)
        # Ensure it's 8-bit for OpenCV
        if image.dtype != np.uint8:
            if image.max() <= 1.01:  # Check if it's normalized 0-1
                image = (image * 255).clip(0, 255)
            else:
                image = image.clip(0, 255)
            image = image.astype(np.uint8)

        # Convert color: RGB -> BGR (OpenCV default)
        if image.ndim == 3:
            if image.shape[-1] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            elif image.shape[-1] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
        cv2.imwrite(str(path), image)
    else:
        raise TypeError(
            f"Expected 'image' to be a tensor or array, "
            f"but got {type(image).__name__}."
        )

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    # Test singleton loader
    foo = ImageLoader()
    bar = ImageLoader()
    print(f"foo: {foo}\nbar: {bar}\nfoo == bar: {foo == bar}")

# endregion
