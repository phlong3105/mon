#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Image Data Structures.

This module provides data structures and utilities for handling image data.
"""

from __future__ import annotations

__all__ = [
    "Frame",
    "Image",
]

from dataclasses import dataclass

import numpy as np
import torch
from numpy import ndarray
from torch import Tensor

from mon.core.path import Path
from mon.core.typing import Int3
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
        path (Path | None, optional): Path to the image file. Defaults to None.
        base_dir (Path | None, optional): Base directory for relative paths.
            This is useful to resolve other files related to the data.
            Defaults to None.
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
            raise TypeError(f"expected image to be a 3D array, "
                            f"got {type(self.image).__name__}.")
        if self.image.ndim != 3:
            raise ValueError(f"expected image to be a 3D array, "
                             f"got {self.image.ndim}D.")
        if is_valid_str(self.path):
            self.path: Path = Path(self.path).normalize()
            if not self.path.is_image_file(exists=True):
                # We only care about the validity of the path, not its existence
                raise ValueError(f"image file not found at {self.path.as_posix()}")
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
    def shape(self) -> Int3:
        """Return the data shape."""
        shape = self.image.shape
        return shape[0], shape[1], shape[2]

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
        path: Path | None = None,
        base_dir: Path | None = None
    ) -> Image:
        """Create an image from a PyTorch tensor.

        Args:
            image (Tensor): Image tensor of shape (1, C, H, W) and values
                ranging from 0.0 to 1.0.
            path (Path | None, optional): Path to the image file. Defaults to None.
            base_dir (Path | None, optional): Base directory for relative paths.
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
            raise ValueError(f"expected image to be a 4D tensor of shape (1, C, H, W), "
                             f"got {image.shape}.")

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
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
