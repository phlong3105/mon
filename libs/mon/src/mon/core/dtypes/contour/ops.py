#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Contour atomic operations.

This module provides atomic operations for contours.
"""

from __future__ import annotations

__all__ = [
    "convert",
    "denormalize",
    "normalize",
]

import numpy as np

from mon.core.enum import BBoxFormat
from .. import image as I


# ==============================================================================
# region CREATION
# ==============================================================================


# endregion


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

def convert(
    contour: np.ndarray,
    fmt    : BBoxFormat,
    imgsz  : tuple[int, int]
) -> np.ndarray:
    """Convert contour points between supported formats.

    Dispatch conversion based on the provided ``fmt``. Supported conversions
    include normalization and denormalization. If the format is not recognized,
    return the input ``contour`` unchanged.

    Args:
        contour: Contour points, formatted as a numpy.ndarray of shape (N, 2).
        fmt: Target conversion format.
        imgsz: Image size as (H, W).

    Returns:
        Converted contour points in the desired format.
    """
    fmt = BBoxFormat(value=fmt)
    match fmt:
        case BBoxFormat.VOC2YOLO:
            return normalize(contour, imgsz)
        case BBoxFormat.YOLO2VOC:
            return denormalize(contour, imgsz)
        case _:
            return contour


# --- Encoding ---


# --- Standardization ---


# --- Structural ---


# --- Statistical ---

def normalize(contour: np.ndarray, imgsz: tuple[int, int]) -> np.ndarray:
    """Normalize contour points to the range [0, 1].

    Convert contour coordinates from pixel units to normalized coordinates
    using the provided ``imgsz``.

    Args:
        contour: Contour points, formatted as a numpy.ndarray of shape (N, 2)
            in pixel coordinates.
        imgsz: Image size as (H, W).

    Returns:
        Normalized contour points, formatted as a numpy.ndarray of shape (N, 2)
        and values ranging from 0.0 to 1.0.
    """
    # Standardize image size
    h, w = I.imgsz(imgsz)

    # Standardize input to (N, 2)
    orig_shape = contour.shape
    contour    = contour.reshape(-1, 2)

    # Vectorized division: [x, y] / [w, h]
    # Adding epsilon 1e-7 prevents division by zero
    scale      = np.array([w, h], dtype=np.float32)
    normalized = contour.astype(np.float32) / (scale + 1e-7)

    return normalized.reshape(orig_shape)


def denormalize(contour: np.ndarray, imgsz: tuple[int, int]) -> np.ndarray:
    """Denormalize contour points from [0, 1] to pixel coordinates.

    Convert contour coordinates from normalized space to pixel coordinates
    using the provided ``imgsz``.

    Args:
        contour: Normalized contour points, formatted as a numpy.ndarray of
            shape (N, 2) and values ranging from 0.0 to 1.0.
        imgsz: Image size as (H, W).

    Returns:
        Denormalized contour points, formatted as a numpy.ndarray of shape
        (N, 2) in pixel coordinates.
    """
    # Standardize image size
    h, w = I.imgsz(imgsz)

    # Standardize input to (N, 2)
    orig_shape   = contour.shape
    contour      = contour.reshape(-1, 2)

    # Vectorized multiplication: [x_norm, y_norm] * [w, h]
    scale        = np.array([w, h], dtype=np.float32)
    denormalized = contour * scale

    return denormalized.reshape(orig_shape)


# --- Geometric ---


# endregion


# ==============================================================================
# region DESTRUCTION
# ==============================================================================


# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
