#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for contour processing functions.

This module provides functions for normalizing, denormalizing, and converting
contour points between different formats.
"""

__all__ = [
    "convert",
    "denormalize",
    "normalize",
]

import numpy as np

from mon.core.enum import BBoxFormat
from .. import image as I


# ----- Normalization -----
def normalize(contour: np.ndarray, imgsz: tuple[int, int]) -> np.ndarray:
    """Normalizes contour points to range [0.0, 1.0].

    Args:
        contour (numpy.ndarray): Contour points as a numpy.ndarray of shape
            (N, 2) in pixel coordinates.
        imgsz (tuple[int, int]): Image size as a tuple of (H, W).

    Returns:
        numpy.ndarray: Normalized contour points as a numpy.ndarray of shape
            (N, 2).
    """
    h0, w0   = I.imgsz(imgsz)
    x, y, *_ = contour.T
    x_norm   = x / w0
    y_norm   = y / h0
    return np.stack((x_norm, y_norm), axis=-1)


def denormalize(contour: np.ndarray, imgsz: tuple[int, int]) -> np.ndarray:
    """Denormalizes contour points from range [0.0, 1.0] to pixel coordinates.

    Args:
        contour (numpy.ndarray): Normalized contour points as a numpy.ndarray of
            shape (N, 2).
        imgsz (tuple[int, int]): Image size as a tuple of (H, W).

    Returns:
        numpy.ndarray: Denormalized contour points as a numpy.ndarray of shape
            (N, 2) in pixel coordinates.
    """
    h0, w0 = I.imgsz(imgsz)
    x_norm, y_norm, *_ = contour.T
    x = x_norm * w0
    y = y_norm * h0
    return np.stack((x, y), axis=-1)


# ----- Conversion -----
def convert(contour: np.ndarray, fmt: BBoxFormat, imgsz: tuple[int, int]) -> np.ndarray:
    """Converts contour points between different formats.
    
    Args:
        contour (numpy.ndarray): Contour points as a numpy.ndarray of shape
            (N, 2).
        fmt (BBoxFormat): The format to convert to.
        imgsz (tuple[int, int]): Image size as a tuple of (H, W).
        
    Returns:
        numpy.ndarray: Converted contour points as a numpy.ndarray of shape
            (N, 2).
    """
    fmt = BBoxFormat(value=fmt)
    match fmt:
        case BBoxFormat.VOC2YOLO:
            return normalize(contour, imgsz)
        case BBoxFormat.YOLO2VOC:
            return denormalize(contour, imgsz)
        case _:
            return contour
