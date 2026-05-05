#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Size and Shape Data Structures.

This module provides data structures and utilities for handling sizes and shapes.
"""

from __future__ import annotations

__all__ = [
    "Size",
]

import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import PIL.Image
import torch
from numpy import ndarray
from torch import Tensor

from mon.core.typing import Int2, IntOrTuple2


# ==============================================================================
# region BASE CLASSES
# ==============================================================================

@dataclass(slots=True, frozen=True)
class Size:
    """Data structure for handling unambiguous image dimensions.

    Attributes:
        height (int): Height of the image.
        width (int): Width of the image.

    Example:
        >>> size0 = Size(height=100, width=200)
        Size(height=100, width=200)
        >>> size0.h
        100
        >>> size0.w
         200
        >>> size0.hw
        (100, 200)
        >>> size0.wh
        (200, 100)
        >>> size0.area
        20000
    """

    height: int
    width: int

    # --- Comparison Operators ---
    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Size):
            return self.height == other.height and self.width == other.width
        return False

    def __lt__(self, other: Any) -> bool:
        if isinstance(other, Size):
            return self.height < other.height or self.width < other.width
        return False

    # --- Container / Sequence Methods ---
    def __getitem__(self, index: int) -> int:
        """Return an item at the given ``index``."""
        if index == 0:
            return self.height
        elif index == 1:
            return self.width
        else:
            raise IndexError(f"index {index} is out of range.")

    # --- Properties ---
    @property
    def h(self):
        """Returns the height."""
        return self.height

    @property
    def w(self):
        """Returns the width."""
        return self.width

    @property
    def hw(self) -> Int2:
        """Returns (Height, Width) for PyTorch and NumPy."""
        return self.height, self.width

    @property
    def wh(self) -> Int2:
        """Returns (Width, Height) for OpenCV and PIL."""
        return self.width, self.height

    @property
    def area(self) -> int:
        return self.height * self.width

    # --- Creation ---
    @classmethod
    def from_tuple(cls, size: IntOrTuple2, fmt: Literal["hw", "wh"] = "hw") -> "Size":
        """Create a new instance from a tuple.

        Args:
            size (IntOrTuple2): Tuple containing the height and width.
            fmt (Literal["hw", "wh"], optional): String describing the format.
                Defaults to "hw".

        Example:
            >>> size0 = Size.from_tuple((100, 200))
            >>> size1 = Size.from_tuple((200, 100), fmt="wh")
        """
        if isinstance(size, (int, float)):
            return cls(height=size, width=size)
        elif fmt.lower() == "hw":
            return cls(height=size[0], width=size[1])
        elif fmt.lower() == "wh":
            return cls(height=size[1], width=size[0])
        raise ValueError(
            f"unsupported format {fmt.lower()}, must be one of ['hw', 'wh']."
        )

    @classmethod
    def from_any(cls, value: Any, divisor: int | None = None) -> "Size":
        """Create a new instance from an arbitrary value.

        Args:
            value (Any): Size-like (i.e., scalar or sequence) or an image.
            divisor (int | None, optional): Divisor size for height and width.
                Defaults to None.

        Example:
            >>> size0 = Size.from_any(100)
            Size(height=100, width=100)
            >>> size1 = Size.from_any((100, 200))
            Size(height=100, width=200)
            >>> size2 = Size.from_any((200, 100), divisor=32)
            Size(height=224, width=128)
            >>> size3 = Size.from_any(torch.rand(1, 3, 224, 224))
            Size(height=224, width=224)
            >>> size4 = Size.from_any(np.random.rand(224, 224, 3))
            Size(height=224, width=224)
            >>> size5 = Size.from_any(PIL.Image.open("path/to/image.jpg"))
            Size(height=..., width=...)
        """
        size = None

        if isinstance(value, Size):
            if divisor:
                size = value.hw
            else:
                return value
        elif isinstance(value, (int, float)):
            # Handle scalars
            size = (int(value), int(value))
        elif isinstance(value, (list, tuple)):
            # Handle Sequences
            if len(value) >= 2:
                # Take the first two elements assuming they represent (H, W)
                size = (value[0], value[1])
            elif len(value) == 1:
                size = (value[0], value[0])
        elif isinstance(value, (Tensor, ndarray)):
            # Handle Tensors/Arrays
            shape = value.shape
            if isinstance(value, Tensor):
                size = (int(shape[-2]), int(shape[-1]))
            else:
                size = (int(shape[0]), int(shape[1]))

        if size is None:
            raise TypeError(f"could not resolve size from {type(value).__name__}.")

        # Apply Divisor (Rounding up to the nearest multiple)
        if divisor:
            h, w = size
            h = int(math.ceil(h / divisor) * divisor)
            w = int(math.ceil(w / divisor) * divisor)
            size = (h, w)

        return cls(height=size[0], width=size[1])

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
