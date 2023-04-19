#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements geometry functions for contours or segments."""

from __future__ import annotations

__all__ = [
    "contour_voc_to_yolo", "contour_yolo_to_voc", "convert_contour",
    "denormalize_contour", "normalize_contour",
]

import numpy as np

from mon.globals import ShapeCode


# region Conversion

def normalize_contour(contour: np.ndarray, height: int, width: int) -> np.ndarray:
    """Normalize contour's points to the range [0.0-1.0]."""
    contour  = contour.copy()
    x, y, *_ = contour.T
    x_norm   = x / width
    y_norm   = y / height
    contour  = np.stack((x_norm, y_norm), axis=-1)
    return contour


def denormalize_contour(contour: np.ndarray, height: int, width: int) -> np.ndarray:
    """Denormalize contour's points."""
    contour = contour.copy()
    x_norm, y_norm, *_ = contour.T
    x       = x_norm * width
    y       = y_norm * height
    contour = np.stack((x, y), axis=-1)
    return contour


contour_voc_to_yolo = normalize_contour
contour_yolo_to_voc = denormalize_contour


def convert_contour(
    contour: np.ndarray,
    code   : ShapeCode | int,
    height : int,
    width  : int
) -> np.ndarray:
    """Convert bounding box."""
    code = ShapeCode.from_value(value=code)
    match code:
        case ShapeCode.VOC2YOLO:
            return contour_voc_to_yolo(contour=contour, height=height, width=width)
        case ShapeCode.YOLO2VOC:
            return contour_yolo_to_voc(contour=contour, height=height, width=width)
        case _:
            return contour
    

# endregion
