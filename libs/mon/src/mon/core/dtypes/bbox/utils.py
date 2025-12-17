#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module of bounding box format validation utilities.

This module provides functions to validate the format of bounding boxes (bboxes)
commonly used in computer vision tasks. It supports various formats including
center-based, corner-based, and normalized bounding boxes.

The default format of a bounding box is: <cx, cy, w, h, a, cls, ...>, where ...
can be any additional information such as confidence score or tracking ID.
For HBBs, the angle ``a`` is always ``0``.
"""

__all__ = [
    "is_cxcywhn",
    "is_normalized",
    "is_xywh",
    "is_xyxy",
]

import numpy as np


# ----- Validation -----
def is_normalized(bbox: np.ndarray) -> bool:
    """Check if a bounding box(es) is normalized.
    
    Args:
        bbox: Bounding box(es) as a numpy.ndarray of shape (4+) or (N, 4+).
    
    Returns:
        True if normalized, False otherwise.
        
    Raises:
        ValueError: If ``bbox`` does not have the correct shape.
    """
    if not (bbox.ndim >= 2 and bbox.shape[-1] >= 4):
        raise ValueError(f"``bbox`` must be of shape (N, 4+), got {bbox.shape}.")
    
    return np.all((bbox[:, :4] >= 0) & (bbox[:, :4] <= 1))


def is_cxcywhn(bbox: np.ndarray, imgsz: tuple[int, int]) -> bool:
    """Check if a bounding box(es) is in CXCYWHN format.
    
    Args:
        bbox: Bounding box(es) as a numpy.ndarray of shape (4+) or (N, 4+).
        imgsz: Image size as a tuple of (H, W).
        
    Returns:
        True if in CXCYWHN format, False otherwise.
        
    Raises:
        ValueError: If ``bbox`` is not of shape (N, 4+).
    """
    if not (bbox.ndim >= 2 and bbox.shape[-1] >= 4):
        raise ValueError(f"``bbox`` must be of shape (N, 4+), got {bbox.shape}.")

    return (
        np.all((bbox[:, :4] >= 0.0) & (bbox[:, :4] <= 1.0))
        and np.all((bbox[:, 2:4] > 0))  # Width and height must be positive
    )


def is_xyxy(bbox: np.ndarray, imgsz: tuple[int, int]) -> bool:
    """Check if a bounding box(es) is in XYXY format.
    
    Args:
        bbox: Bounding box(es) as a numpy.ndarray of shape (4+) or (N, 4+).
        imgsz: Image size as a tuple of (H, W).
    
    Returns:
        True if in XYXY format, False otherwise.
        
    Raises:
        ValueError: If ``bbox`` is not of shape (N, 4+).
    """
    if not (bbox.ndim >= 2 and bbox.shape[-1] >= 4):
        raise ValueError(f"``bbox`` must be of shape (N, 4+), got {bbox.shape}.")

    if is_cxcywhn(bbox, imgsz):
        return False

    # Extract first bbox for format checking
    x, y, w, h = bbox[0, :4]
    if w > x and h > y:  # VOC: x_max > x_min, y_max > y_min
        return True
    else:
        return False


def is_xywh(bbox: np.ndarray, imgsz: tuple[int, int]) -> bool:
    """Check if a bounding box(es) is in XYWH format.
    
    Args:
        bbox: Bounding box(es) as a numpy.ndarray of shape (4+) or (N, 4+).
        imgsz: Image size as a tuple of (H, W).
        
    Returns:
        True if in XYWH format, False otherwise.
        
    Raises:
        ValueError: If ``bbox`` is not of shape (N, 4+).
    """
    if not (bbox.ndim >= 2 and bbox.shape[-1] >= 4):
        raise ValueError(f"``bbox`` must be of shape (N, 4+), got {bbox.shape}.")

    if is_cxcywhn(bbox, imgsz):
        return False

    # Extract first bbox for format checking
    x, y, w, h = bbox[0, :4]
    if w + x > x and h + y > y:  # VOC: w=x_max, h=y_max, so x_min+w > x_min
        return True  # COCO: w=width, h=height
    else:
        return False
