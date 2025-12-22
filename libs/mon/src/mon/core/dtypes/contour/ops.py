#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Contour atomic operations.

This module provides pure functions that perform a single mathematical or
structural change to the contour data.
"""

__all__ = [
    "convert",
    "denormalize",
    "normalize",
]

import numpy as np

from mon.core.enum import BBoxFormat
from .. import image as I


# ==============================================================================
# VALIDATION & SANITIZATION (Integrity Checks)
# ==============================================================================

# --- Verify (Schema and range checking) ---


# --- Clean (Fixing corrupt values/nulls) ---


# ==============================================================================
# GEOMETRIC TRANSFORMATIONS (Resizing, Warping)
# ==============================================================================

# --- Analytics (Area, Perimeter, Centroid calculations) ---


# --- Metrics ---


# --- Project (Affine, Perspective, and Coordinate space transforms) ---
def convert(contour: np.ndarray, fmt: BBoxFormat, imgsz: tuple[int, int]) -> np.ndarray:
    """Convert contour points between supported formats.

    Dispatch conversion based on the provided BBoxFormat. Supported conversions
    include normalization and denormalization. If the format is not recognized,
    return the input contour unchanged.

    Args:
        contour: Contour points as a numpy.ndarray of shape (N, 2).
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


# --- Reshape (Resize, Crop, Padding) ---


# ==============================================================================
# STATISTICAL OPERATIONS (Normalization, Scaling)
# ==============================================================================

# --- Normalize (Mean/Std, Min-Max scaling) ---
def normalize(contour: np.ndarray, imgsz: tuple[int, int]) -> np.ndarray:
    """Normalize contour points to the range [0, 1].

    Convert contour coordinates from pixel units to normalized coordinates
    using the provided image size.

    Args:
        contour: Contour points as a numpy.ndarray of shape (N, 2) in pixel
            coordinates.
        imgsz: Image size as (H, W).

    Returns:
        Normalized contour points a numpy.ndarray of shape (N, 2) with values
        in the range [0, 1].
    """
    h0, w0   = I.imgsz(imgsz)
    x, y, *_ = contour.T
    x_norm   = x / w0
    y_norm   = y / h0
    return np.stack((x_norm, y_norm), axis=-1)


def denormalize(contour: np.ndarray, imgsz: tuple[int, int]) -> np.ndarray:
    """Denormalize contour points from [0, 1] to pixel coordinates.

    Convert contour coordinates from normalized space to pixel coordinates
    using the provided image size.

    Args:
        contour: Normalized contour points as a numpy.ndarray with shape (N, 2)
            with values in the range [0, 1].
        imgsz: Image size as (H, W).

    Returns:
        Denormalized contour points in pixel coordinates.
    """
    h0, w0 = I.imgsz(imgsz)
    x_n, y_n, *_ = contour.T
    x = x_n * w0
    y = y_n * h0
    return np.stack((x, y), axis=-1)
