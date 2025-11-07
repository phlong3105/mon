#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements processing function for contour data type.

Common Tasks:
    - Format conversions.
"""

__all__ = [
    "convert",
    "denormalize",
    "normalize",
]

import numpy as np

from mon.core.dtypes import image as I
from mon.core.enum import BBoxFormat


# ----- Normalization -----
def normalize(contour: np.ndarray, imgsz: tuple[int, int]) -> np.ndarray:
    """Normalize contour points in :math:`[0.0, 1.0]`.

    Args:
        contour: Contour points as a ``numpy.ndarray`` of shape :math:`(N, 2)`.
        imgsz: Image size as a ``tuple`` of :math:`(H, W)`.

    Returns:
        Normalized contour points as a ``numpy.ndarray`` of shape :math:`(N, 2)`.
    """
    h, w     = I.imgsz(imgsz)
    x, y, *_ = contour.T
    x_norm   = x / w
    y_norm   = y / h
    return np.stack((x_norm, y_norm), axis=-1)


def denormalize(contour: np.ndarray, imgsz: tuple[int, int]) -> np.ndarray:
    """Denormalize contour points to pixel coordinates.

    Args:
        contour: Normalized points as a ``numpy.ndarray``  of shape :math:`(N, 2)`
            in range :math:`[0.0, 1.0]`.
        imgsz: Image size as a ``tuple`` of :math:`(H, W)`.

    Returns:
        Denormalized contour points as a ``numpy.ndarray`` of shape :math:`(N, 2)`.
    """
    h, w = I.imgsz(imgsz)
    x_norm, y_norm, *_ = contour.T
    x    = x_norm * w
    y    = y_norm * h
    return np.stack((x, y), axis=-1)


# ----- Conversion -----
def convert(contour: np.ndarray, fmt: BBoxFormat, imgsz: tuple[int, int]) -> np.ndarray:
    """Convert contour."""
    fmt = BBoxFormat.from_value(value=fmt)
    match fmt:
        case BBoxFormat.VOC2YOLO:
            return normalize(contour, imgsz)
        case BBoxFormat.YOLO2VOC:
            return denormalize(contour, imgsz)
        case _:
            return contour
