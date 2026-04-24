#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Image Patching Operations.

This module provides operations for patching images.
"""

from __future__ import annotations

__all__ = [
    "HannWindowImagePatcher",
    "ImagePatcher",
    "NaiveGridImagePatcher",
    "UniformAveragingImagePatcher",
]

from typing import Iterator

import torch
from tensordict import TensorDict
from torch import Tensor
from torch.nn import functional as F

from mon.core import PATCHERS


# ==============================================================================
# region BASES CLASSES
# ==============================================================================

class ImagePatcher:
    """Base class for stitching several patches together into a single image."""

    # We use a small epsilon to avoid division by zero during normalization
    eps = 1e-8

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        image: Tensor,
        size: int = 512,
        stride: int | None = 384,
        name: str = "hann_window",
        *args, **kwargs
    ):
        """Initialize a new instance.

        Args:
            name (str): Name of the patcher.
            image (Tensor): Original image tensor of shape (B, C, H, W) and
                values ranging from 0.0 to 1.0. This is used to allocate the
                output tensor and compute the number of patches.
            size (int, optional): Size of the patches to stitch together.
                Defaults to 512.
            stride (int, optional): Stride between patches. Defaults to None
                means non-overlapping patches.
        """
        # Assign attributes
        self._name = name
        self._size = size
        self._stride = stride or size

        # Allocate resources
        self._image: Tensor = None
        self._image_padded: Tensor = None
        self._canvases: dict[str, Tensor] = {}
        self._weight_masks: dict[str, Tensor] = {}
        self._window: Tensor = None
        self._output: TensorDict = None

        self.setup(image=image)

    def setup(self, image: Tensor):
        """Setup the patcher for a new image.

        Args:
            image (Tensor): Original image tensor of shape (B, C, H, W) and
                values ranging from 0.0 to 1.0. This is used to allocate the
                output tensor and compute the number of patches.
        """
        s = self._size
        step = self._stride
        b, c, h, w = image.shape
        device = image.device

        # 1. Pad the image so we don't drop the bottom/right edges
        pad_h = (step - (h - s) % step) % step
        pad_w = (step - (w - s) % step) % step
        image_padded = F.pad(image, (0, pad_w, 0, pad_h), mode="reflect")

        # 3. Setup output buffers (accumulator and weight map for blending
        self._image = image
        self._image_padded = image_padded
        self._canvases: dict[str, Tensor] = {}
        self._weight_masks: dict[str, Tensor] = {}
        self._window = self._get_window()
        self._output = None

    def _get_window(self) -> Tensor:
        """Get the window for blending patches."""
        s = self._size
        device = self._image.device

        if self._name == "hann_window":
            window_1d = torch.hann_window(s).to(device)
            window_2d = window_1d.unsqueeze(0) * window_1d.unsqueeze(1)
            window = window_2d.unsqueeze(0).unsqueeze(0).repeat(1, 1, 1, 1)
        else:
            window = torch.ones((1, 1, s, s), device=device)
        return window

    # --- Container / Sequence Methods ---
    def __iter__(self) -> Iterator:
        """Iterates over the image and yields patches with their coordinates."""
        b, c, h, w = self._image.shape
        s = self._size
        step = self._stride

        for y in range(0, h - s + 1, step):
            for x in range(0, w - s + 1, step):
                patch = self._image_padded[:, :, y:y+s, x:x+s]
                yield patch, x, y

    # --- Callable & Context Manager ---
    def __call__(self, patches: TensorDict | dict[str, Tensor], x: int, y: int):
        """Accumulate results back into the patcher.

        Args:
            patches (TensorDict | dict[str, Tensor]): Input patches tensors of
                shape (B, C, H, W) and values ranging from 0.0 to 1.0.
            x (int): X-coordinate of the top-left corner of the patch in the
                original image.
            y (int): Y-coordinate of the top-left corner of the patch in the
                original image.
        """
        s = self._size
        for key, patch in patches.items():
            if key not in self._canvases:
                b, c, _, _ = patch.shape
                _, _, h, w = self._image_padded.shape
                self._canvases[key] = torch.zeros((b, c, h, w), device=patch.device)
                self._weight_masks[key] = torch.zeros((b, c, h, w), device=patch.device)

            self._canvases[key][:, :, y:y+s, x:x+s] += patch * self._window
            self._weight_masks[key][:, :, y:y+s, x:x+s] += self._window

    # --- Properties ---
    @property
    def output(self) -> TensorDict:
        """Return the stitched image tensor."""
        if self._output is None:
            _, _, h, w = self._image.shape
            outputs = {}
            for key, canvas in self._canvases.items():
                weight_mask = self._weight_masks[key]
                canvas = canvas / (weight_mask + self.eps)
                canvas = canvas[:, :, :h, :w]
                outputs[key] = canvas
            self._output = TensorDict(outputs, batch_size=[])

        return self._output

# endregion


# ==============================================================================
# region CONCRETE IMPLEMENTATIONS
# ==============================================================================

@PATCHERS.register(name="naive_grid")
class NaiveGridImagePatcher(ImagePatcher):
    """Non-Overlapping Grid Stitching (Naive Cropping).

    This is the absolute simplest method. The image is chopped into a rigid
    grid (e.g., exactly adjacent 512x512 blocks). Each block is passed through
    the model independently and then concatenated back together like tiles on
    a floor.

    - Pros: Extremely fast. Zero redundant computation.
    - Cons: Produces the most brutal, visible "checkerboard" seams imaginable.
      Because convolutional padding behaves differently at the edge of a patch
      than in the center, the exact boundary pixels will never match their
      neighbors.
    """

    # --- Lifecycle & Initialization ---
    def __init__(self, image: Tensor, size: int = 512, *args, **kwargs):
        """Initialize a new instance.

        Args:
            image (Tensor): Original image tensor of shape (B, C, H, W) and
                values ranging from 0.0 to 1.0. This is used to allocate the
                output tensor and compute the number of patches.
            size (int, optional): Size of the patches to stitch together.
                Defaults to 512.
        """
        super().__init__(
            name="naive_grid",
            image=image,
            size=size,
            stride=None,
            *args, **kwargs
        )


@PATCHERS.register(name="uniform")
class UniformAveragingImagePatcher(ImagePatcher):
    """Overlapping with Uniform Averaging.

    This method is a simple extension of the non-overlapping grid. The patches
    are still arranged in a rigid grid, but they overlap with each other by a
    fixed amount (e.g., 128 pixels). The overlapping regions are blended
    together by averaging the pixel values.

    - Pros: Blurs out the harsh grid lines of the naive method.
    - Cons: Causes "ghosting" or blurriness at the boundaries. If the model
      enhances a structural edge (like a text line) slightly differently in
      Patch A vs. Patch B, averaging them creates a double-vision blurry effect.
    """

    # --- Lifecycle & Initialization ---
    def __init__(self, image: Tensor, size: int = 512, stride: int = 384, *args, **kwargs):
        """Initialize a new instance.

        Args:
            image (Tensor): Original image tensor of shape (B, C, H, W) and
                values ranging from 0.0 to 1.0. This is used to allocate the
                output tensor and compute the number of patches.
            size (int, optional): Size of the patches to stitch together.
                Defaults to 512.
            stride (int, optional): Stride between patches. Defaults to 384 for
                128 pixels of overlap with 512x512 patches.
        """
        super().__init__(
            name="uniform",
            image=image,
            size=size,
            stride=stride,
            *args, **kwargs
        )


@PATCHERS.register(name="hann_window")
class HannWindowImagePatcher(ImagePatcher):
    """Overlapping with Weighted Windowing (Gaussian / Hann).

    This method uses a sliding window, but applies a mathematical weight mask
    (like a 2D Bell Curve or Hann window) to every patch. The center pixels of
    the patch are given a weight of $1.0$, and the weights gracefully taper off
    to 0.0 at the edges.

    - Pros: Almost completely eliminates visible border seams. The transitions
      between patches are mathematically smoothed.
    - Cons: It still suffers from Global Illumination Inconsistency. While the
      seams are smooth, you will still see large "blobs" of different brightness
      levels across a 4K image because the model's global context is restricted
      to the patch size.
    """

    # --- Lifecycle & Initialization ---
    def __init__(self, image: Tensor, size: int = 512, stride: int = 384, *args, **kwargs):
        """Initialize a new instance.

        Args:
            image (Tensor): Original image tensor of shape (B, C, H, W) and
                values ranging from 0.0 to 1.0. This is used to allocate the
                output tensor and compute the number of patches.
            size (int, optional): Size of the patches to stitch together.
                Defaults to 512.
            stride (int, optional): Stride between patches. Defaults to 384 for
                128 pixels of overlap with 512x512 patches.
        """
        super().__init__(
            name="hann_window",
            image=image,
            size=size,
            stride=stride,
            *args, **kwargs
        )

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
