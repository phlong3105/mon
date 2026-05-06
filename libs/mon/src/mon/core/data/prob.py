#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Probabilities Data Structures.

This module provides data structures and utilities for handling probabilities.
"""

from __future__ import annotations

__all__ = [
    "Prob",
]

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy import ndarray

from .data import Data


# ==============================================================================
# region BASE CLASSES
# ==============================================================================

@dataclass
class Prob(Data):
    """Data structure for handling probabilities (i.e., class scores).

    Attributes:
        prob (ndarray): Probability vector of shape (``num_classes``).
    """

    prob: ndarray

    # --- Lifecycle & Initialization ---
    def __post_init__(self):
        """Perform post-initialization tasks.

        Raises:
            TypeError: If ``probs`` is not an ndarray.
            ValueError: If ``probs`` is not a 1D array.
        """
        # Validate inputs
        if not isinstance(self.prob, ndarray):
            raise TypeError(f"expected prob to be an array, "
                            f"got {type(self.prob).__name__}.")
        if self.prob.ndim != 1:
            raise ValueError(f"expected prob to be a 1D array, got {self.prob.ndim}D.")

    # --- Representation ---
    def __len__(self) -> int:
        """Return the length of the container."""
        return len(self.prob)

    def __getitem__(self, index: int) -> float:
        """Return the element at the given ``index``."""
        return float(self.prob[index])

    # --- Properties ---
    @property
    def data(self) -> ndarray:
        """Return the underlying data."""
        return self.prob

    @property
    def shape(self) -> int:
        """Return the data shape."""
        return len(self.prob)

    @property
    def num_classes(self) -> int:
        """Return the number of classes."""
        return len(self.prob)

    @property
    def top1_idx(self) -> int:
        """Return the index of the top-1 class."""
        return int(np.argmax(self.prob))

    @property
    def top1(self) -> float:
        """Return the score of the top-1 class."""
        return float(np.max(self.data))

    @property
    def top5_idxes(self) -> list[int]:
        """Return the indices of the top-5 classes."""
        return list(np.argsort(self.prob)[-5:][::-1])

    @property
    def top5(self) -> ndarray:
        """Return the confidence scores of the top-5 classes."""
        return self.prob[self.top5_idxes]

    @property
    def meta(self) -> dict[str, Any]:
        """Return metadata describing the data."""
        return {
            "shape": self.shape,
            "num_classes": self.num_classes,
            "top1_idx": self.top1_idx,
            "top1": self.top1,
            "top5_idxes": self.top5_idxes,
            "top5": self.top5,
        }

    # --- Creation ---
    @classmethod
    def from_class_id(cls, class_id: int, num_classes: int) -> "Prob":
        """Create a new instance from a class ID and the total number of classes.

        Convert a class ID to a one-hot encoded probability vector.

        Args:
            class_id (int): Class ID.
            num_classes (int): Total number of classes.

        Returns:
            Probs: An one-hot encoded probability vector of shape
                (``num_classes``) where the index corresponding to ``class_id``
                is 1.0 and all other indices are 0.0.

        Raises:
            ValueError: If ``num_classes`` is not positive or if ``class_id``
                is out of range.
        """
        # Validate inputs
        if num_classes <= 0:
            raise ValueError(f"expected num_classes to be a positive integer, "
                             f"got {num_classes}.")
        if not (0 <= class_id < num_classes):
            raise ValueError(f"expected class_id in range [0, {num_classes}), "
                             f"got {class_id}.")

        prob = np.zeros(num_classes, dtype=np.float32)
        prob[class_id] = 1.0
        return cls(prob=prob)

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
