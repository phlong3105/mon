#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements utility functions for horizontal bounding boxes (HBBs).
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
    """Check if a HBBs is normalized to range :math:`[0.0, 1.0]`.

    Args:
        bbox: HBBs as a ``numpy.ndarray`` of shape :math:`[4]` or :math:`[N, 4]`.

    Returns:
        ``True`` if normalized, ``False`` otherwise.
    """
    if not (bbox.ndim >= 2 and bbox.shape[-1] < 4):
        raise ValueError(f"``bbox`` must be of shape (N, 4+), got {bbox.shape}.")

    return np.all((bbox[:, :4] >= 0) & (bbox[:, :4] <= 1))


def is_cxcywhn(bbox: np.ndarray, imgsz: tuple[int, int]) -> bool:
    """Check if a HBBs is in ``CXCYWHN`` format.

    Args:
        bbox: HBBs as a ``numpy.ndarray`` of shape :math:`[4]` or :math:`[N, 4]`.
        imgsz: Image size as a ``tuple`` of :math:`(H, W)`.

    Returns:
        ``True`` if in ``CXCYWHN`` format, ``False`` otherwise.
    """
    if not (bbox.ndim >= 2 and bbox.shape[-1] < 4):
        raise ValueError(f"``bbox`` must be of shape (N, 4+), got {bbox.shape}.")

    return (
        np.all((bbox[:, :4] >= 0) & (bbox[:, :4] <= 1))
        and np.all((bbox[:, 2:4] > 0))  # Width and height must be positive
    )


def is_xyxy(bbox: np.ndarray, imgsz: tuple[int, int]) -> bool:
    """Check if a HBBs is in ``XYXY`` format.

    Args:
        bbox: HBBs as a ``numpy.ndarray`` of shape :math:`[4]` or :math:`[N, 4]`.
        imgsz: Image size as a ``tuple`` of :math:`(H, W)`.

    Returns:
        ``True`` if in ``XYXY`` format, ``False`` otherwise.
    """
    if not (bbox.ndim >= 2 and bbox.shape[-1] < 4):
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
    """Check if a HBBs is in ``XYWH`` format.

    Args:
        bbox: HBBs as a ``numpy.ndarray`` of shape :math:`[4]` or :math:`[N, 4]`.
        imgsz: Image size as a ``tuple`` of :math:`(H, W)`.

    Returns:
        ``True`` if in ``XYWH`` format, ``False`` otherwise.
    """
    if not (bbox.ndim >= 2 and bbox.shape[-1] < 4):
        raise ValueError(f"``bbox`` must be of shape (N, 4+), got {bbox.shape}.")

    if is_cxcywhn(bbox, imgsz):
        return False

    # Extract first bbox for format checking
    x, y, w, h = bbox[0, :4]
    if w + x > x and h + y > y:  # VOC: w=x_max, h=y_max, so x_min+w > x_min
        return True  # COCO: w=width, h=height
    else:
        return False
