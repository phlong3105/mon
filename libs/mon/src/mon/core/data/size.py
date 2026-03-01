#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Size and Shape Data Structures.

This module provides data structures and utilities for handling sizes and shapes.
"""

from __future__ import annotations

__all__ = [
    "Size",
    "SizeLike",
]

import math
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias, Union

from numpy import ndarray
from torch import Tensor

from mon.core.typing import IntOrTuple2, TensorOrArray


# ==============================================================================
# region BASE CLASSES
# ==============================================================================

@dataclass(slots=True, frozen=True)
class Size:
    """Data structure for handling unambiguous image dimensions.

    Attributes:
        height (int): Height of the image.
        width (int): Width of the image.
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
            raise IndexError(f"Index {index} is out of range.")

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
    def hw(self) -> tuple[int, int]:
        """Returns (Height, Width) for PyTorch and NumPy."""
        return self.height, self.width

    @property
    def wh(self) -> tuple[int, int]:
        """Returns (Width, Height) for OpenCV and PIL."""
        return self.width, self.height

    @property
    def area(self) -> int:
        return self.height * self.width

    # --- Creation ---
    @classmethod
    def from_tuple(cls, size: IntOrTuple2, format: Literal["hw", "wh"] = "hw") -> "Size":
        """Create a new instance from a tuple.

        Args:
            size (IntOrTuple2): Tuple containing the height and width.
            format (Literal["hw", "wh"], optional): String describing the format.
                Defaults to "hw".
        """
        if isinstance(size, (int, float)):
            return cls(height=size, width=size)
        elif format.lower() == "hw":
            return cls(height=size[0], width=size[1])
        elif format.lower() == "wh":
            return cls(height=size[1], width=size[0])
        raise ValueError(f"Expected format 'hw' or 'wh', got '{format}'.")

    @classmethod
    def from_value(cls, value: Any, divisor: int | None = None) -> "Size":
        """Create a new instance from an arbitrary value.

        Args:
            value (Any): Size-like (i.e., scalar or sequence) or an image.
            divisor (int, optional): Divisor size for height and width.
                Defaults to None.
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
                size = (int(shape[-3]), int(shape[-2]))

        if size is None:
            raise TypeError(f"Could not get size from {type(value).__name__}.")

        # Apply Divisor (Rounding up to the nearest multiple)
        if divisor:
            h, w = size
            h = int(math.ceil(h / divisor) * divisor)
            w = int(math.ceil(w / divisor) * divisor)
            size = (h, w)

        return cls(height=size[0], width=size[1])

# endregion


# ==============================================================================
# region TYPE DEFINITIONS
# ==============================================================================

SizeLike: TypeAlias = Union[Size, IntOrTuple2, TensorOrArray]


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
