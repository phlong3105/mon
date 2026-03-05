#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Image Data Structures.

This module provides data structures and utilities for handling image data.
"""

from __future__ import annotations

__all__ = [
    "Frame",
    "Image",
    "parse_image_shape",
    "parse_imgsz",
    "to_image_array",
    "to_image_tensor",
]

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from numpy import ndarray
from torch import Tensor

from mon.core.path import Path
from mon.core.typing import Int3, PathLike, TensorOrArray
from mon.core.utils import is_valid_str
from .data import Data
from .size import Size


# ==============================================================================
# region BASE CLASSES
# ==============================================================================

@dataclass
class Image(Data):
    """Data structure for handling a single image.

    Attributes:
        image (ndarray): RGB or grayscale image array of shape (H, W, C) and
            values ranging from 0 to 255.
        path (Path, optional): Path to the image file. Defaults to None.
        base_dir (Path, optional): Base directory for relative paths. This is
            useful to resolve other files related to the data. Defaults to None.
    """

    image: ndarray
    path: Path | None = None
    base_dir: Path | None = None

    # --- Lifecycle & Initialization ---
    def __post_init__(self):
        """Perform post-initialization tasks.

        Raises:
            TypeError: If ``image`` is not an array.
            ValueError: If ``image`` is not a 3D array.
        """
        # Validate inputs
        if not isinstance(self.image, ndarray):
            raise TypeError(
                f"Expected 'image' to be an array, "
                f"but got '{type(self.image).__name__}'."
            )
        if self.image.ndim != 3:
            raise ValueError(
                f"Expected 'image' to be a 3D array, "
                f"but got {self.image.ndim}D array."
            )
        if is_valid_str(self.path):
            self.path = Path(self.path).normalize()
            if not self.path.is_image_file(exists=True):
                # We only care about the validity of the path, not its existence
                raise ValueError(f"Image file not found at: {self.path}")
        if is_valid_str(self.base_dir):
            self.base_dir = Path(self.base_dir).normalize()

    # --- Representation ---
    def __len__(self) -> int:
        """Return the length of the container."""
        return 1

    def __getitem__(self, index: int) -> ndarray:
        """Return the element at the given ``index``."""
        return self.image

    # --- Properties ---
    @property
    def data(self) -> ndarray:
        """Return the underlying data."""
        return self.image

    @property
    def shape(self) -> tuple[int, int, int]:
        """Return the data shape."""
        return self.image.shape

    @property
    def imgsz(self) -> Size:
        """Return the image size as (H, W)."""
        return Size(height=self.shape[0], width=self.shape[1])

    @property
    def num_channels(self) -> int:
        """Return the number of image channels."""
        return int(self.shape[-1])

    @property
    def hash(self) -> int | None:
        """Return the hash of the image file if ``path`` is available."""
        return self.path.hash if isinstance(self.path, Path) else None

    @property
    def meta(self) -> dict:
        """Return metadata describing the data."""
        return {
            "path": self.path,
            "base_dir": self.base_dir,
            "shape": self.shape,
            "imgsz": self.imgsz,
            "hash": self.hash,
        }

    @property
    def is_color(self) -> bool:
        """Return True if the image is color, False otherwise."""
        return self.num_channels in [3, 4]

    @property
    def is_grayscale(self) -> bool:
        """Return True if the image is grayscale, False otherwise."""
        return self.num_channels == 1

    @property
    def center(self) -> ndarray:
        """Return the image center of shape (H / 2, W / 2)."""
        h, w = self.imgsz
        return np.array([h // 2, w // 2], dtype=np.float32)

    # --- Creation ---
    @classmethod
    def from_tensor(
        cls,
        image: Tensor,
        path: PathLike | None = None,
        base_dir: PathLike | None = None
    ) -> Image:
        """Create an image from a PyTorch tensor.

        Args:
            image (Tensor): Image tensor of shape (1, C, H, W) and values
                ranging from 0.0 to 1.0.
            path (PathLike, optional): Path to the image file. Defaults to None.
            base_dir (Path, optional): Base directory for relative paths.
                This is useful to resolve other files related to the data.
                Defaults to None.

        Returns:
            Image: An Image instance containing the image data.

        Raises:
            ValueError: If ``image`` is not a 4D tensor of shape (1, C, H, W)
                with C in [1, 3, 4].
        """
        # Validate inputs
        if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] not in [1, 3, 4]:
            raise ValueError(
                f"Expected 'image' to be a 4D tensor of shape (1, C, H, W), "
                f"but got {image.shape}."
            )

        image = image[0].detach().cpu()
        # [C, H, W] -> [H, W, C]
        image = image.permute(1, 2, 0).clamp(0, 1).mul(255).round().to(torch.uint8)
        image = image.numpy()
        return cls(image=image, path=path, base_dir=base_dir)

    # --- Transformation ---
    def to_tensor(self, normalize: bool = False) -> Tensor:
        """Return the image as a PyTorch tensor.

        Args:
            normalize (bool, optional), If True, scales pixel values to range
                [0.0, 1.0]. Defaults to False.

        Returns:
            Tensor: An image tensor of shape (1, C, H, W) and values ranging
                from 0.0 to 1.0 if ``normalize`` is True, else ranging from
                0 to 255.
        """
        # Convert to tensor and permute: [H, W, C] -> [C, H, W]
        tensor = torch.from_numpy(self.image).permute(2, 0, 1).float()

        if normalize:
            tensor = tensor.div(255.0)

        # Add batch dimension: [1, C, H, W]
        return tensor.unsqueeze(0).contiguous()

    def pad_square(self, value: int = 0, mode: str = "constant") -> ndarray:
        """Pad the image to make it square.

        Args:
            value (int, optional): Padding value. Defaults to 0.
            mode (str, optional): Padding mode. Defaults to "constant".

        Returns:
            ndarray: Padded square image of shape (S, S, C), where S is the
                maximum of (H, W).
        """
        h, w = self.imgsz
        s = max(h, w)

        # Calculate padding for top, bottom, left, right
        pad_h = s - h
        pad_w = s - w

        top, bottom = int(pad_h // 2), int(pad_h - (pad_h // 2))
        left, right = int(pad_w // 2), int(pad_w - (pad_w // 2))

        # Construction of padding width tuple
        # For (H, W, C): ((top, bottom), (left, right), (0, 0))
        pad_width = [(top, bottom), (left, right)]
        if self.num_channels == 3:
            pad_width.append((0, 0))  # Don't pad the channel dimension

        return np.pad(self.image, pad_width, mode=mode, constant_values=value)


@dataclass
class Frame(Image):
    """Data structure for handling a frame within a video.

    Attributes:
        index (int, optional): Frame index within the video. Defaults to -1.
    """

    index: int = -1

    # --- Properties ---
    @property
    def frame_path(self) -> Path | None:
        """Construct a path for the frame based on the video path and index.

        Return the stored ``path`` if no video path is provided.
        """
        if self.path is not None:
            path = self.path
            return path.parent / path.stem / f"{path.stem}_{self.index}.jpg"
        else:
            return self.path

    @property
    def meta(self) -> dict:
        """Return metadata describing the ``data``."""
        return {
            "path": self.frame_path,
            "video_path": self.path,
            "base_dir": self.base_dir,
            "index": self.index,
            "shape": self.shape,
            "imgsz": self.imgsz,
            "hash": self.hash,
        }

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

    raise ValueError(f"Could not get shape from {type(image)}.")


def parse_imgsz(value: Any, divisor: int = None) -> Size:
    """Extract the size of an image as a tuple of (H, W).

    Args:
        value (Any): Size-like (i.e., scalar or sequence) or an image.
        divisor (int, optional): Divisor size for height and width.
            Defaults to None.

    Returns:
        Size: Image size as (H, W).
    """
    return Size.from_value(value=value, divisor=divisor)


# --- Selection ---


# --- Aggregation ---

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

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
