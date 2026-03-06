#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Bounding Box Operation.

This module provides operations for bounding boxes.
"""

from __future__ import annotations

__all__ = [
    "to_2d_bbox",
]

import numpy as np
from numpy import ndarray


# ==============================================================================
# region VALIDATION
# ==============================================================================

# endregion


# ==============================================================================
# region RETRIEVAL
# ==============================================================================

# --- Accessing ---


# --- Selection ---


# --- Aggregation ---

# endregion


# ==============================================================================
# region MUTATION
# ==============================================================================

# --- Alternation ---


# --- Rearrangement ---


# --- Addition ---


# --- Removal ---

# endregion


# ==============================================================================
# region COMPUTATION
# ==============================================================================

# --- Arithmetic ---


# --- Comparison ---


# --- Logical ---


# --- Geometric ---

# endregion


# ==============================================================================
# region TRANSFORMATION
# ==============================================================================

# --- Casting ---


# --- Encoding ---


# --- Standardization ---


# --- Structural ---

def to_2d_bbox(bbox: ndarray | list | tuple) -> ndarray:
    """Standardize bounding boxes as a 2D array.

    Args:
        bbox (ndarray | list | tuple): Bounding boxes to standardize. Can be a
            single or a batch of bounding boxes.

    Returns:
        ndarray: Standardized bounding boxes of shape (N, M).

    Raises:
        ValueError: If ``bbox``'s type is unsupported.
        TypeError: If list/tuple elements have inconsistent shapes.
    """
    # Convert a list or tuple into an array
    if isinstance(bbox, (list, tuple)):
        try:
            bbox = np.array(bbox, dtype=np.float32)
        except ValueError:
            # Handle jagged arrays (e.g., one box has 7 elements, another has 8)
            raise ValueError(
                "Expected all elements in 'bbox' to have the same shape."
            )

    # Validate inputs
    if not isinstance(bbox, ndarray):
        raise TypeError(
            f"Expected 'bbox' to be an array, but got {type(bbox).__name__}."
        )

    # Handle various shapes
    if bbox.ndim == 1:
        return bbox[np.newaxis, :]   # [5+] -> [1, 5+]
    elif bbox.ndim == 3:
        return np.squeeze(bbox)      # [1, N, 5+] -> [N, 5+]

    return bbox


# --- Statistical ---


# --- Geometric ---

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
