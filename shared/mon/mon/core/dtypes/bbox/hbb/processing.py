#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements processing functions for HBB data type.

Common Tasks:
    - Format conversions.
    - Transformations.
"""

__all__ = [
    "area",
    "center",
    "center_distance",
    "ciou",
    "coco_to_voc",
    "coco_to_yolo",
    "convert",
    "corners",
    "corners_pts",
    "crop_center",
    "cxcywhn_to_xywh",
    "cxcywhn_to_xyxy",
    "denormalize",
    "diou",
    "enclosing",
    "filter_iou",
    "giou",
    "iou",
    "iou_matrix",
    "normalize",
    "pad_square",
    "split",
    "to_2d",
    "voc_to_coco",
    "voc_to_yolo",
    "xywh_to_cxcywhn",
    "xywh_to_xyxy",
    "xyxy_to_cxcywhn",
    "xyxy_to_xywh",
    "yolo_to_coco",
    "yolo_to_voc",
]

import math
from typing import Union

import cv2
import numpy as np

from mon.core.dtypes import image as I
from mon.core.enum import BBoxFormat
from .utils import is_normalized


# ----- IoU -----
def iou(bbox1: np.ndarray, bbox2: np.ndarray) -> np.ndarray:
    """Compute IoU between two sets of HBBs.

    Args:
        bbox1: HBBs as a ``numpy.ndarray`` of shape :math:`(4+)` or :math:`(N, 4+)`
            in ``XYXY`` format.
        bbox2: HBBs as a ``numpy.ndarray`` of shape :math:`(4+)` or :math:`[M, 4+]`
            in ``XYXY`` format.

    Returns:
        Pairwise IoU values as a ``numpy.ndarray`` of shape :math:`[N, M]`.

    Raises:
        ValueError: If ``bbox1`` or ``bbox2`` is not 1D or 2D.
    """
    # Ensure 2D arrays
    bbox1 = to_2d(bbox1)
    bbox2 = to_2d(bbox2)

    # Expand the dimensions of the bboxes to calculate pairwise IoU values.
    bbox1 = np.expand_dims(bbox1, 1)
    bbox2 = np.expand_dims(bbox2, 0)
    
    # IoU calculation.
    xx1 = np.maximum(bbox1[..., 0], bbox2[..., 0])
    yy1 = np.maximum(bbox1[..., 1], bbox2[..., 1])
    xx2 = np.minimum(bbox1[..., 2], bbox2[..., 2])
    yy2 = np.minimum(bbox1[..., 3], bbox2[..., 3])
    
    # Intersection area
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    wh = w * h
    
    # Union area
    union = ((bbox1[..., 2] - bbox1[..., 0]) * (bbox1[..., 3] - bbox1[..., 1])
             + (bbox2[..., 2] - bbox2[..., 0]) * (bbox2[..., 3] - bbox2[..., 1]) - wh)
    return wh / union


def giou(bbox1: np.ndarray, bbox2: np.ndarray) -> np.ndarray:
    """Compute generalized IoU between two sets of boxes.

    Args:
        bbox1: HBBs as a ``numpy.ndarray`` of shape :math:`(4+)` or :math:`(N, 4+)`
            in ``XYXY`` format.
        bbox2: HBBs as a ``numpy.ndarray`` of shape :math:`(4+)` or :math:`[M, 4+]`
            in ``XYXY`` format.

    Returns:
        Pairwise GIoU values as a ``numpy.ndarray`` of shape :math:`[N, M]`.

    Raises:
        ValueError: If ``bbox1`` or ``bbox2`` is not 1D or 2D.

    References:
        - Paper: https://arxiv.org/pdf/1902.09630.pdf
    """
    # Ensure 2D arrays
    bbox1 = to_2d(bbox1)
    bbox2 = to_2d(bbox2)
    
    # Expand dimensions for pairwise computation
    bbox1 = np.expand_dims(bbox1, 1)
    bbox2 = np.expand_dims(bbox2, 0)

    # Intersection coordinates
    xx1 = np.maximum(bbox1[..., 0], bbox2[..., 0])
    yy1 = np.maximum(bbox1[..., 1], bbox2[..., 1])
    xx2 = np.minimum(bbox1[..., 2], bbox2[..., 2])
    yy2 = np.minimum(bbox1[..., 3], bbox2[..., 3])

    # Intersection area
    w   = np.maximum(0.0, xx2 - xx1)
    h   = np.maximum(0.0, yy2 - yy1)
    wh  = w * h

    # Union area
    union = (
        (bbox1[..., 2] - bbox1[..., 0]) * (bbox1[..., 3] - bbox1[..., 1]) +
        (bbox2[..., 2] - bbox2[..., 0]) * (bbox2[..., 3] - bbox2[..., 1]) - wh
    )

    # IoU
    iou_ = wh / union

    # Enclosing box coordinates
    xxc1 = np.minimum(bbox1[..., 0], bbox2[..., 0])
    yyc1 = np.minimum(bbox1[..., 1], bbox2[..., 1])
    xxc2 = np.maximum(bbox1[..., 2], bbox2[..., 2])
    yyc2 = np.maximum(bbox1[..., 3], bbox2[..., 3])

    # Enclosing area
    wc   = xxc2 - xxc1
    hc   = yyc2 - yyc1
    area_enclose = wc * hc

    # GIoU
    giou_ = iou_ - (area_enclose - union) / area_enclose
    # giou_ = (giou_ + 1.0) / 2.0  # Commented out: GIoU typically in [-1, 1], not [0, 1]
    return giou_


def diou(bbox1: np.ndarray, bbox2: np.ndarray) -> np.ndarray:
    """Compute distance IoU between two sets of boxes.

    Args:
        bbox1: HBBs as a ``numpy.ndarray`` of shape :math:`(4+)` or :math:`(N, 4+)`
            in ``XYXY`` format.
        bbox2: HBBs as a ``numpy.ndarray`` of shape :math:`(4+)` or :math:`[M, 4+]`
            in ``XYXY`` format.

    Returns:
        Pairwise DIoU values as a ``numpy.ndarray`` of shape :math:`[N, M]`.

    Raises:
        ValueError: If ``bbox1`` or ``bbox2`` is not 1D or 2D.

    References:
        - Paper: https://arxiv.org/pdf/1902.09630.pdf
    """
    # Ensure 2D arrays
    bbox1 = to_2d(bbox1)
    bbox2 = to_2d(bbox2)

    # Expand dimensions for pairwise computation
    bbox1 = np.expand_dims(bbox1, 1)
    bbox2 = np.expand_dims(bbox2, 0)

    # Intersection coordinates
    xx1 = np.maximum(bbox1[..., 0], bbox2[..., 0])
    yy1 = np.maximum(bbox1[..., 1], bbox2[..., 1])
    xx2 = np.minimum(bbox1[..., 2], bbox2[..., 2])
    yy2 = np.minimum(bbox1[..., 3], bbox2[..., 3])

    # Intersection area
    w   = np.maximum(0.0, xx2 - xx1)
    h   = np.maximum(0.0, yy2 - yy1)
    wh  = w * h

    # Union area
    union = (
        (bbox1[..., 2] - bbox1[..., 0]) * (bbox1[..., 3] - bbox1[..., 1]) +
        (bbox2[..., 2] - bbox2[..., 0]) * (bbox2[..., 3] - bbox2[..., 1]) - wh
    )

    # IoU
    iou_ = wh / union

    # Center distances
    cx1 = (bbox1[..., 0] + bbox1[..., 2]) / 2.0
    cy1 = (bbox1[..., 1] + bbox1[..., 3]) / 2.0
    cx2 = (bbox2[..., 0] + bbox2[..., 2]) / 2.0
    cy2 = (bbox2[..., 1] + bbox2[..., 3]) / 2.0
    inner_diag = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2

    # Enclosing box diagonal
    xxc1 = np.minimum(bbox1[..., 0], bbox2[..., 0])
    yyc1 = np.minimum(bbox1[..., 1], bbox2[..., 1])
    xxc2 = np.maximum(bbox1[..., 2], bbox2[..., 2])
    yyc2 = np.maximum(bbox1[..., 3], bbox2[..., 3])
    outer_diag = (xxc2 - xxc1) ** 2 + (yyc2 - yyc1) ** 2

    # DIoU
    diou_ = iou_ - inner_diag / outer_diag
    # diou_ = (diou_ + 1) / 2.0  # Commented: DIoU typically in [-1, 1], not [0, 1]
    return diou_


def ciou(bbox1: np.ndarray, bbox2: np.ndarray) -> np.ndarray:
    """Compute complete IoU between two sets of boxes.

    Args:
        bbox1: HBBs as a ``numpy.ndarray`` of shape :math:`(4+)` or :math:`(N, 4+)`
            in ``XYXY`` format.
        bbox2: HBBs as a ``numpy.ndarray`` of shape :math:`(4+)` or :math:`[M, 4+]`
            in ``XYXY`` format.

    Returns:
        Pairwise CIoU values as a ``numpy.ndarray`` of shape :math:`[N, M]`.

    Raises:
        ValueError: If ``bbox1`` or ``bbox2`` is not 1D or 2D.

    References:
        - Paper: https://arxiv.org/pdf/1902.09630.pdf
    """
    # Ensure 2D arrays
    bbox1 = to_2d(bbox1)
    bbox2 = to_2d(bbox2)

    # Expand dimensions for pairwise computation
    bbox1 = np.expand_dims(bbox1, 1)
    bbox2 = np.expand_dims(bbox2, 0)

    # Intersection coordinates
    xx1 = np.maximum(bbox1[..., 0], bbox2[..., 0])
    yy1 = np.maximum(bbox1[..., 1], bbox2[..., 1])
    xx2 = np.minimum(bbox1[..., 2], bbox2[..., 2])
    yy2 = np.minimum(bbox1[..., 3], bbox2[..., 3])

    # Intersection area
    w   = np.maximum(0.0, xx2 - xx1)
    h   = np.maximum(0.0, yy2 - yy1)
    wh  = w * h

    # Union area
    union = (
        (bbox1[..., 2] - bbox1[..., 0]) * (bbox1[..., 3] - bbox1[..., 1]) +
        (bbox2[..., 2] - bbox2[..., 0]) * (bbox2[..., 3] - bbox2[..., 1]) - wh
    )

    # IoU
    iou_ = wh / union

    # Center distances
    cx1 = (bbox1[..., 0] + bbox1[..., 2]) / 2.0
    cy1 = (bbox1[..., 1] + bbox1[..., 3]) / 2.0
    cx2 = (bbox2[..., 0] + bbox2[..., 2]) / 2.0
    cy2 = (bbox2[..., 1] + bbox2[..., 3]) / 2.0
    inner_diag = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2

    # Enclosing box diagonal
    xxc1 = np.minimum(bbox1[..., 0], bbox2[..., 0])
    yyc1 = np.minimum(bbox1[..., 1], bbox2[..., 1])
    xxc2 = np.maximum(bbox1[..., 2], bbox2[..., 2])
    yyc2 = np.maximum(bbox1[..., 3], bbox2[..., 3])
    outer_diag = (xxc2 - xxc1) ** 2 + (yyc2 - yyc1) ** 2

    # Aspect ratio term
    w1 = bbox1[..., 2] - bbox1[..., 0]
    h1 = bbox1[..., 3] - bbox1[..., 1]
    w2 = bbox2[..., 2] - bbox2[..., 0]
    h2 = bbox2[..., 3] - bbox2[..., 1]
    h2 += 1.0  # Prevent division by zero
    h1 += 1.0  # Prevent division by zero
    arctan = np.arctan(w2 / h2) - np.arctan(w1 / h1)
    v      = (4 / (np.pi ** 2)) * (arctan ** 2)
    S      = 1 - iou
    alpha  = v / (S + v)

    # CIoU
    ciou_ = iou_ - inner_diag / outer_diag - alpha * v
    # ciou_ = (ciou_ + 1) / 2.0  # Commented: CIoU typically in [-1, 1], not [0, 1]
    return ciou_


def iou_matrix(bbox: np.ndarray) -> np.ndarray:
    """Calculate pairwise IoU for all pairs of HBBs using matrix operations.

    Args:
        bbox: HBBs as a ``numpy.ndarray`` of shape :math:`(N, 4+)` in ``XYXY`` format.

    Returns:
        Pairwise IoU matrix as a ``numpy.ndarray`` of shape :math:`[N, N]` where
        the element :math:`(i, j)` is IoU between boxes :math:`i` and :math:`j`.
    """
    # Ensure 2D arrays
    bbox = to_2d(bbox)
    N    = bbox.shape[0]

    # Extract coordinates
    x1 = bbox[:, 0:1]  # Shape (N, 1)
    y1 = bbox[:, 1:2]
    x2 = bbox[:, 2:3]
    y2 = bbox[:, 3:4]

    # Compute intersection coordinates
    x_left   = np.maximum(x1, x1.T)  # Shape (N, N)
    y_top    = np.maximum(y1, y1.T)
    x_right  = np.minimum(x2, x2.T)
    y_bottom = np.minimum(y2, y2.T)

    # Intersection area
    intersection = np.maximum(0, x_right - x_left) * np.maximum(0, y_bottom - y_top)

    # Box areas
    areas = (x2 - x1) * (y2 - y1)  # Shape (N, 1)
    union = areas + areas.T - intersection

    # Avoid division by zero
    iou_matrix = np.where(union > 0, intersection / union, 0)

    # Set diagonal to 0 (no self-IoU)
    np.fill_diagonal(iou_matrix, 0)

    return iou_matrix


def filter_iou(bbox: np.ndarray, iou_thres: float = 0.5) -> np.ndarray:
    """Filter HBBs that <= IoU threshold.

    Args:
        bbox: HBBs as a ``numpy.ndarray`` of shape :math:`(N, 4+)` in ``XYXY`` format.
        iou_thres: IoU threshold for filtering. Default: ``0.5``.

    Returns:
        Filtered HBBs as a ``numpy.ndarray`` of shape :math:`(N, 4+)` in ``XYXY`` format.
    """
    # Calculate IoU matrix
    matrix = iou_matrix(bbox)

    # Initialize keep mask
    N     = len(bbox)
    keep  = np.ones(N, dtype=bool)
    areas = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])

    # Filter boxes based on IoU
    for i in range(N):
        if not keep[i]:
            continue
        # Find boxes with high IoU
        high_iou = matrix[i] >= iou_thres
        # Compare areas to decide which to keep
        for j in np.where(high_iou)[0]:
            if keep[j] and areas[i] <= areas[j]:
                keep[i] = False
                break
            else:
                keep[j] = False

    return bbox[keep]


# ----- Properties -----
def center_distance(bbox1: np.ndarray, bbox2: np.ndarray) -> np.ndarray:
    """Measure center distance(s) between two sets of boxes.

    Args:
        bbox1: HBBs as a ``numpy.ndarray`` of shape :math:`(4+)` or :math:`(N, 4+)`
            in ``XYXY`` format.
        bbox2: HBBs as a ``numpy.ndarray`` of shape :math:`(4+)` or :math:`[M, 4+]`
            in ``XYXY`` format.

    Returns:
        Pairwise center distances as a ``numpy.ndarray`` of shape :math:`[N, M]`.

    Raises:
        ValueError: If ``bbox1`` or ``bbox2`` is not 1D or 2D.

    Notes:
        Coarse implementation, not recommended alone for association due to instability.
    """
    # Ensure 2D arrays
    bbox1 = to_2d(bbox1)
    bbox2 = to_2d(bbox2)

    # Expand dimensions for pairwise computation
    bbox1 = np.expand_dims(bbox1, 1)
    bbox2 = np.expand_dims(bbox2, 0)

    # Center coordinates
    cx1 = (bbox1[..., 0] + bbox1[..., 2]) / 2.0  # Fixed: Use bbox1 only
    cy1 = (bbox1[..., 1] + bbox1[..., 3]) / 2.0  # Fixed: Use bbox1 only
    cx2 = (bbox2[..., 0] + bbox2[..., 2]) / 2.0  # Fixed: Use bbox2 only
    cy2 = (bbox2[..., 1] + bbox2[..., 3]) / 2.0  # Fixed: Use bbox2 only

    # Squared Euclidean distance
    ct_dist2 = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2

    # Euclidean distance
    ct_dist = np.sqrt(ct_dist2)

    # Normalize and invert to [0, 1] (smaller distance = higher value)
    ct_dist_max = np.max(ct_dist)
    if ct_dist_max > 0:  # Avoid division by zero
        ct_dist = ct_dist / ct_dist_max
        ct_dist = ct_dist_max - ct_dist  # Invert: max distance = 0, min = max
    else:
        ct_dist = np.ones_like(ct_dist)  # All distances 0 -> all 1

    return ct_dist


def area(bbox: np.ndarray) -> np.ndarray:
    """Compute area of HBBs.

    Args:
        bbox: HBBs as a ``numpy.ndarray`` of shape :math:`(4+)` or :math:`(N, 4+)`
            in ``XYXY`` format.

    Returns:
        Area(s) as a ``numpy.ndarray`` of shape :math:`[1]` or :math:`[N]` shape.

    Raises:
        ValueError: If ``bbox`` is not 1D or 2D.
    """
    bbox = to_2d(bbox)
    x1   = bbox[..., 0]
    y1   = bbox[..., 1]
    x2   = bbox[..., 2]
    y2   = bbox[..., 3]
    return (x2 - x1) * (y2 - y1)


def center(bbox: np.ndarray) -> np.ndarray:
    """Compute center(s) of HBBs.

    Args:
        bbox: HBBs as a ``numpy.ndarray`` of shape :math:`(4+)` or :math:`(N, 4+)`
            in ``XYXY`` format.

    Returns:
        Center(s) as a ``numpy.ndarray`` of shape :math:`[1, 2]` or :math:`[N, 2]`,
        :math:`[cx, cy]` format.

    Raises:
        ValueError: If ``bbox`` is not 1D or 2D.
    """
    bbox = to_2d(bbox)
    x1   = bbox[..., 0]
    y1   = bbox[..., 1]
    x2   = bbox[..., 2]
    y2   = bbox[..., 3]
    cx   = x1 + (x2 - x1) / 2.0
    cy   = y1 + (y2 - y1) / 2.0
    return np.stack((cx, cy), -1)


def corners(bbox: np.ndarray) -> np.ndarray:
    """Get corner(s) of HBBs.

    Args:
        bbox: HBBs as a ``numpy.ndarray`` of shape :math:`(4+)` or :math:`(N, 4+)`
            in ``XYXY`` format.

    Returns:
        Corners as a ``numpy.ndarray`` of shape :math:`[N, 8]` each element is of
        shape :math:`[x1, y1, x2, y2, x3, y3, x4, y4]`.

    Raises:
        ValueError: If ``bbox`` is not 1D or 2D.
    """
    bbox = to_2d(bbox)
    x1   = bbox[..., 0]
    y1   = bbox[..., 1]
    x2   = bbox[..., 2]
    y2   = bbox[..., 3]
    w    = x2 - x1
    h    = y2 - y1
    c_x1 = x1
    c_y1 = y1
    c_x2 = x1 + w
    c_y2 = y1
    c_x3 = x2
    c_y3 = y2
    c_x4 = x1
    c_y4 = y1 + h
    return np.hstack((c_x1, c_y1, c_x2, c_y2, c_x3, c_y3, c_x4, c_y4))


def corners_pts(bbox: np.ndarray) -> np.ndarray:
    """Get corner(s) of HBBs as points.

    Args:
        bbox: HBBs as a ``numpy.ndarray`` in :math:`(4+)` or :math:`(N, 4+)` in
            ``XYXY`` format.

    Returns:
        Corners as a ``numpy.ndarray`` of shape :math:`[N, 4, 2]` where each element
        is of shape :math:`[[x1, y1], [x2, y2], [x3, y3], [x4, y4]]`.

    Raises:
        ValueError: If ``bbox`` is not 1D or 2D.
    """
    bbox = to_2d(bbox)
    x1   = bbox[..., 0]
    y1   = bbox[..., 1]
    x2   = bbox[..., 2]
    y2   = bbox[..., 3]
    w    = x2 - x1
    h    = y2 - y1
    c_x1 = x1
    c_y1 = y1
    c_x2 = x1 + w
    c_y2 = y1
    c_x3 = x2
    c_y3 = y2
    c_x4 = x1
    c_y4 = y1 + h
    return np.array([[c_x1, c_y1], [c_x2, c_y2], [c_x3, c_y3], [c_x4, c_y4]], np.int32)


def enclosing(bbox: np.ndarray) -> np.ndarray:
    """Get enclosing box(es) for rotated corners.

    Args:
        bbox: HBBs as a ``numpy.ndarray`` of shape
            :math:`[..., 8], [x1, y1, x2, y2, x3, y3, x4, y4]` format.

    Returns:
        HBBs as a ``numpy.ndarray`` of shape :math:`[..., 4]` in XYXY format.

    Raises:
        ValueError: If ``bbox`` last dimension is not ``8``.
    """
    if bbox.shape[-1] < 8:
        raise ValueError(f"``bbox`` last dimension must be 8, got {bbox.shape[-1]}.")
    x_ = bbox[:, [0, 2, 4, 6]]
    y_ = bbox[:, [1, 3, 5, 7]]
    x1 = np.min(x_, 1).reshape(-1, 1)
    y1 = np.min(y_, 1).reshape(-1, 1)
    x2 = np.max(x_, 1).reshape(-1, 1)
    y2 = np.max(y_, 1).reshape(-1, 1)
    return np.hstack((x1, y1, x2, y2, bbox[:, 8:]))


# ----- Resizing -----
def crop_center(image: np.ndarray, bbox: np.ndarray, imgsz: int) -> tuple[np.ndarray, np.ndarray]:
    """Center crop an image with HBBs.

    Args:
        image: Image as a ``numpy.ndarray`` of shape :math:`(H, W, C)`.
        bbox: HBBs as a ``numpy.ndarray`` of shape :math:`(N, 4+)` in ``CXCYWHN`` format.
        imgsz: Target size as a tuple of :math:`(H, W)` or a single ``int`` for square crops.
    """
    h0, w0 = I.imgsz(image)
    h1, w1 = I.imgsz(imgsz)
    if h1 > h0 or w1 > w0:
        raise ValueError(f"Target size {imgsz} exceeds original image size {image.shape[:2]}.")
    
    # Calculate crop region (center of image)
    x_start = max(0, (w0 - w1) // 2)
    y_start = max(0, (h0 - h1) // 2)
    x_end   = x_start + w1
    y_end   = y_start + h1

    # Crop the image
    cropped_image = image[y_start:y_end, x_start:x_end].copy()

    # Adjust bounding box
    bbox = convert(bbox, fmt=BBoxFormat.CXCYWHN2XYXY, imgsz=(h0, w0))
    adjusted_bbox = []
    for b in bbox:
        x1, y1, x2, y2 = b[0:4]
        # Shift coordinates relative to crop top-left
        x1 = x1 - x_start
        y1 = y1 - y_start
        x2 = x2 - x_start
        y2 = y2 - y_start
        # Check if bbox is within crop (allow partial overlap)
        if x2 <= 0 or y2 <= 0 or x1 >= w1 or y1 >= h1:
            continue  # Bbox is completely outside crop
        # Clip coordinates to crop boundaries
        x1 = max(0, min(x1, w1))
        y1 = max(0, min(y1, h1))
        x2 = max(0, min(x2, w1))
        y2 = max(0, min(y2, h1))
        # Skip if bbox is invalid (zero or negative size)
        if x1 >= x2 or y1 >= y2:
            continue
        adjusted_bbox.append(np.concatenate(([x1, y1, x2, y2], b[4:])))
    adjusted_bbox = np.array(adjusted_bbox, np.float32)
    adjusted_bbox = convert(adjusted_bbox, fmt=BBoxFormat.XYXY2CXCYWHN, imgsz=(h1, w1))
    return cropped_image, adjusted_bbox


def pad_square(image: np.ndarray, bbox: np.ndarray, pad_value: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Pad an image with HBBs to make it square.
    
    Args:
        image: Image as a ``numpy.ndarray`` of shape :math:`(H, W, C)`.
        bbox: HBBs as a ``numpy.ndarray`` of shape :math:`(N, 4+)` in ``CXCYWHN`` format.
        pad_value: Padding value. Default: ``0``.

    Returns:
        Padded image and adjusted HBBs.
    """
    h0, w0 = I.imgsz(image)
    dim    = max(h0, w0)
    pad_h  = (dim - h0) // 2
    pad_w  = (dim - w0) // 2

    # Pad image
    padded_image = cv2.copyMakeBorder(
        src        = image,
        top        = pad_h,
        bottom     = dim - h0 - pad_h,
        left       = pad_w,
        right      = dim - w0 - pad_w,
        borderType = cv2.BORDER_CONSTANT,
        value      = [pad_value, pad_value, pad_value],
    )
    
    # Adjust bounding boxes
    bbox = convert(bbox, fmt=BBoxFormat.CXCYWHN2XYXY, imgsz=(h0, w0))
    adjusted_bbox = []
    for b in bbox:
        x1, y1, x2, y2 = b[0:4]
        x1 += pad_w
        x2 += pad_w
        y1 += pad_h
        y2 += pad_h
        adjusted_bbox.append(np.concatenate(([x1, y1, x2, y2], b[4:])))
    adjusted_bbox = np.array(adjusted_bbox, np.float32)
    adjusted_bbox = convert(adjusted_bbox, fmt=BBoxFormat.XYXY2CXCYWHN, imgsz=(dim, dim))
    return padded_image, adjusted_bbox


def split(image: np.ndarray, bbox: np.ndarray, n: int = 2) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Split an image with HBBs into ``n`` equal parts.

    Args:
        image: Image as a ``numpy.ndarray`` of shape :math:`(H, W, C)`.
        bbox: HBBs as a ``numpy.ndarray`` of shape :math:`(N, 4+)` in ``CXCYWHN`` format.
        n: Number of parts to split into (positive integer). Default: ``2``.

    Raises:
        ValueError: If inputs are invalid (e.g., image shape, bboxes, n).
    """
    if not isinstance(image, np.ndarray) or len(image.shape) != 3:
        raise ValueError(f"``image`` must be a numpy.ndarray of shape (H, W, C), got {image.shape}.")
    if not isinstance(bbox, np.ndarray)  or bbox.ndim != 2 or bbox.shape[1] < 4:
        raise ValueError(f"``bboxes`` must be a numpy.ndarray of shape (N, M) with M >= 4, got {bbox.shape}.")
    if n < 1:
        raise ValueError(f"``n`` must be a positive integer, got {n}.")

    h, w = I.imgsz(image)
    if n > h * w:
        raise ValueError(f"``n`` ({n}) exceeds image pixel count ({h * w}).")

    # Determine orientation
    is_portrait = h > w

    # Determine rows and cols
    if n == 1:
        rows, cols = 1, 1
    elif n == 2:
        # Explicitly set grid for N=2 based on orientation
        rows = 2 if is_portrait else 1
        cols = 1 if is_portrait else 2
    else:
        # General case: start with approximate square grid
        rows = math.ceil(math.sqrt(n))
        cols = math.ceil(n / rows)
        # Adjust to ensure rows * cols = n, prioritizing orientation
        candidates = []
        for r in range(1, n + 1):
            c = math.ceil(n / r)
            if r * c == n:
                candidates.append((r, c))
        if not candidates:
            raise ValueError(f"Cannot find valid rows and cols for n={n}")
        # Select grid based on orientation
        if is_portrait:
            # Prefer more rows (taller sub-images)
            rows, cols = max(candidates, key=lambda x: x[0] / x[1])
        else:
            # Prefer more cols (wider sub-images)
            rows, cols = max(candidates, key=lambda x: x[1] / x[0])

    # Compute sub-images and adjust bboxes
    sub_h      = h // rows
    sub_w      = w // cols
    sub_images = []
    sub_bboxes = []

    for i in range(rows):
        for j in range(cols):
            if len(sub_images) >= n:
                break
            # Compute sub-image boundaries
            y_start   = i * sub_h
            y_end     = min((i + 1) * sub_h, h)
            x_start   = j * sub_w
            x_end     = min((j + 1) * sub_w, w)
            sub_image = image[y_start:y_end, x_start:x_end]
            if sub_image.size == 0:
                continue
            sub_images.append(sub_image)

            # Adjust bboxes
            sub_bboxes_i     = []
            sub_h_i, sub_w_i = sub_image.shape[:2]
            for b in bbox:
                cx_n, cy_n, w_n, h_n = b[:4]
                cx = cx_n * w
                cy = cy_n * h
                x1 = cx - w_n * w / 2
                x2 = cx + w_n * w / 2
                y1 = cy - h_n * h / 2
                y2 = cy + h_n * h / 2

                # Check if bbox intersects sub-image
                if x2 > x_start and x1 < x_end and y2 > y_start and y1 < y_end:
                    # Compute new bbox in sub-image coordinates
                    x1_new   = max(x1, x_start) - x_start
                    x2_new   = min(x2, x_end)   - x_start
                    y1_new   = max(y1, y_start) - y_start
                    y2_new   = min(y2, y_end)   - y_start
                    cx_n_new = (x1_new + x2_new) / 2 / sub_w_i
                    cy_n_new = (y1_new + y2_new) / 2 / sub_h_i
                    w_n_new  = (x2_new - x1_new) / sub_w_i
                    h_n_new  = (y2_new - y1_new) / sub_h_i
                    if w_n_new > 0 and h_n_new > 0:
                        bbox_new = np.concatenate(([cx_n_new, cy_n_new, w_n_new, h_n_new], b[4:]))
                        sub_bboxes_i.append(bbox_new)
            sub_bboxes.append(np.array(sub_bboxes_i) if sub_bboxes_i else np.zeros((0, bbox.shape[1]), dtype=np.float32))

    # Pad with empty sub-images/bboxes if needed
    while len(sub_images) < n:
        sub_images.append(np.zeros_like(sub_images[0]))
        sub_bboxes.append(np.zeros((0, bbox.shape[1]), dtype=np.float32))

    return sub_images, sub_bboxes


# ----- Normalization -----
def normalize(bbox: np.ndarray, imgsz: tuple[int, int]) -> np.ndarray:
    """Normalize HBBs according to image dimensions.

    Args:
        bbox: HBBs as a ``numpy.ndarray`` of shape :math:`(4+)` or :math:`(N, 4+)`.
        imgsz: Image size as a ``tuple`` of :math:`(H, W)`.
    """
    height, width = I.imgsz(imgsz)
    bbox = to_2d(bbox)
    if is_normalized(bbox):
        return bbox

    b1, b2, b3, b4, *rest = bbox.T
    b1 = b1 / width
    b2 = b2 / height
    b3 = b3 / width
    b4 = b4 / height
    return np.stack((b1, b2, b3, b4, *rest), axis=-1)


def denormalize(bbox: np.ndarray, imgsz: tuple[int, int]) -> np.ndarray:
    """Denormalize HBBs according to image dimensions.

    Args:
        bbox: HBBs as a ``numpy.ndarray`` of shape :math:`(4+)` or :math:`(N, 4+)`.
        imgsz: Image size as a ``tuple`` of :math:`(H, W)`.
    """
    height, width = I.imgsz(imgsz)
    bbox = to_2d(bbox)
    if not is_normalized(bbox):
        return bbox

    b1, b2, b3, b4, *rest = bbox.T
    b1 = b1 * width
    b2 = b2 * height
    b3 = b3 * width
    b4 = b4 * height
    return np.stack((b1, b2, b3, b4, *rest), axis=-1)


# ----- Shape Conversion -----
def to_2d(bbox: Union[np.ndarray, list, tuple]) -> np.ndarray:
    """Convert a 1D, 2D, or 3D box(es) to 2D.

    Args:
        bbox: HBBs as a ``numpy.ndarray``, ``list``, or ``tuple`` of shape
            :math:`(4+)` or :math:`(N, 4+)`.

    Returns:
        HBBs as a ``numpy.ndarray`` of shape :math:`(N, 4+)`.
    """
    if isinstance(bbox, np.ndarray):
        if bbox.ndim == 1:                                                      # [4+]
            bbox = np.expand_dims(bbox, axis=0)                                 # [4+]       -> [1, 4+]
        elif bbox.ndim == 3 and bbox.shape[0] == 1:                             # [1, N, 4+]
            bbox = np.squeeze(bbox, axis=0)                                     # [1, N, 4+] -> [N, 4+]
    elif isinstance(bbox, list | tuple):
        bbox = np.array(bbox, dtype=np.float32)
        if bbox[0].ndim == 1:                                                   # [[4+], ...]
            bbox = np.stack(bbox, axis=0)                                       # [[4+], ...]    -> [N, 4+]
        elif bbox[0].ndim == 2:                                                 # [[N, 4+], ...]
            bbox = np.concatenate(bbox, axis=0)                                 # [[N, 4+], ...] -> [N*, 4+]
        else:
            raise TypeError(f"``bbox`` list/tuple must contain consistent 1D or 2D "
                            f"numpy.ndarray, got mixed types or dimensions: "
                            f"{[type(b) for b in bbox]} "
                            f"{[b.shape for b in bbox if b is not None]}.")
    else:
        raise ValueError(f"``bbox`` must be a numpy.ndarray, or list/tuple, got {type(bbox)}.")

    return bbox


# ----- Format Conversion -----
def xywh_to_cxcywhn(bbox: np.ndarray, imgsz: tuple[int, int]) -> np.ndarray:
    """Convert boxes from ``XYWH`` to ``CXCYWHN`` format.

    Args:
        bbox: HBBs as a ``numpy.ndarray`` of shape :math:`(4+)` or :math:`(N, 4+)`
            in ``XYWH`` format.
        imgsz: Image size as a ``tuple`` of :math:`(H, W)`.

    Returns:
        HBBs as a ``numpy.ndarray`` of shape :math:`(N, 4+)` in ``CXCYWHN`` format.
    """
    height, width = I.imgsz(imgsz)
    bbox = to_2d(bbox)
    x, y, w, h, *rest = bbox.T
    cx   = x + (w / 2.0)
    cy   = y + (h / 2.0)
    cx_n = cx / width
    cy_n = cy / height
    w_n  = w  / width
    h_n  = h  / height
    return np.stack((cx_n, cy_n, w_n, h_n, *rest), axis=-1)


def xywh_to_xyxy(bbox: np.ndarray, imgsz: tuple[int, int]) -> np.ndarray:
    """Convert boxes from ``XYWH`` to ``XYXY`` format.

    Args:
        bbox: HBBs as a ``numpy.ndarray`` of shape :math:`(4+)` or :math:`(N, 4+)`
            in ``XYWH`` format.
        imgsz: Image size as a ``tuple`` of :math:`(H, W)`.

    Returns:
        HBBs as a ``numpy.ndarray`` of shape :math:`(N, 4+)` in ``XYXY`` format.
    """
    bbox = to_2d(bbox)
    x, y, w, h, *rest = bbox.T
    x2 = x + w
    y2 = y + h
    return np.stack((x, y, x2, y2, *rest), axis=-1)


def xyxy_to_cxcywhn(bbox: np.ndarray, imgsz: tuple[int, int]) -> np.ndarray:
    """Convert boxes from ``XYXY`` to ``CXCYWHN`` format.

    Args:
        bbox: HBBs as a ``numpy.ndarray`` of shape :math:`(4+)` or :math:`(N, 4+)`
            in ``XYXY`` format.
        imgsz: Image size as a ``tuple`` of :math:`(H, W)`.

    Returns:
        HBBs as a ``numpy.ndarray`` of shape :math:`(N, 4+)` in ``CXCYWHN`` format.
    """
    height, width = I.imgsz(imgsz)
    bbox = to_2d(bbox)
    x1, y1, x2, y2, *rest = bbox.T
    w    = x2 - x1
    h    = y2 - y1
    cx   = x1 + (w / 2.0)
    cy   = y1 + (h / 2.0)
    cx_n = cx / width
    cy_n = cy / height
    w_n  = w  / width
    h_n  = h  / height
    return np.stack((cx_n, cy_n, w_n, h_n, *rest), axis=-1)


def xyxy_to_xywh(bbox: np.ndarray, imgsz: tuple[int, int]) -> np.ndarray:
    """Convert boxes from ``XYXY`` to ``XYWH`` format.

   Args:
        bbox: HBBs as a ``numpy.ndarray`` of shape :math:`(4+)` or :math:`(N, 4+)`
            in ``XYXY`` format.
        imgsz: Image size as a ``tuple`` of :math:`(H, W)`.

    Returns:
        HBBs as a ``numpy.ndarray`` of shape :math:`(N, 4+)` in ``XYWH`` format.
    """
    bbox = to_2d(bbox)
    x1, y1, x2, y2, *rest = bbox.T
    w = x2 - x1
    h = y2 - y1
    return np.stack((x1, y1, w, h, *rest), axis=-1)


def cxcywhn_to_xywh(bbox: np.ndarray, imgsz: tuple[int, int]) -> np.ndarray:
    """Convert boxes from ``CXCYWHN`` to ``XYWH`` format.

    Args:
        bbox: HBBs as a ``numpy.ndarray`` of shape :math:`(4+)` or :math:`(N, 4+)`
            in ``CXCYWHN`` format.
        imgsz: Image size as a ``tuple`` of :math:`(H, W)`.

    Returns:
        HBBs as a ``numpy.ndarray`` of shape :math:`(N, 4+)` in ``XYWH`` format.
    """
    height, width = I.imgsz(imgsz)
    bbox = to_2d(bbox)
    cx_n, cy_n, w_n, h_n, *rest = bbox.T
    w = w_n * width
    h = h_n * height
    x = (cx_n * width)  - (w / 2.0)
    y = (cy_n * height) - (h / 2.0)
    # Combine processed columns with rest
    return np.stack((x, y, w, h, *rest), axis=-1)


def cxcywhn_to_xyxy(bbox: np.ndarray, imgsz: tuple[int, int]) -> np.ndarray:
    """Convert boxes from ``CXCYWHN`` to ``XYXY`` format.

    Args:
        bbox: HBBs as a ``numpy.ndarray`` of shape :math:`(4+)` or :math:`(N, 4+)`
            in ``CXCYWHN`` format.
        imgsz: Image size as a ``tuple`` of :math:`(H, W)`.

    Returns:
        HBBs as a ``numpy.ndarray`` of shape :math:`(N, 4+)` in ``XYXY`` format.
    """
    height, width = I.imgsz(imgsz)
    bbox = to_2d(bbox)
    cx_n, cy_n, w_n, h_n, *rest = bbox.T
    x1 = width  * (cx_n - w_n / 2)
    y1 = height * (cy_n - h_n / 2)
    x2 = width  * (cx_n + w_n / 2)
    y2 = height * (cy_n + h_n / 2)
    return np.stack((x1, y1, x2, y2, *rest), axis=-1)


coco_to_voc  = xywh_to_xyxy
coco_to_yolo = xywh_to_cxcywhn
voc_to_coco  = xyxy_to_xywh
voc_to_yolo  = xyxy_to_cxcywhn
yolo_to_coco = cxcywhn_to_xywh
yolo_to_voc  = cxcywhn_to_xyxy


def convert(bbox: np.ndarray, fmt: BBoxFormat, imgsz: tuple[int, int]) -> np.ndarray:
    """Convert HBBs between formats.

    Args:
        bbox: HBBs as a ``numpy.ndarray`` of shape :math:`(N, 4+)`.
        fmt: Conversion code as ``BBoxFormat`` or ``int``.
        imgsz: Image size as a ``tuple`` of :math:`(H, W)`.

    Returns:
        HBBs as a ``numpy.ndarray`` of shape :math:`(N, 4+)`, output format varied
        by ``fmt``.

    Raises:
        ValueError: If ``fmt`` is invalid.
    """
    if len(bbox) == 0:
        return bbox

    fmt = BBoxFormat.from_value(value=fmt)
    if fmt in BBoxFormat.formats():
        return bbox
    match fmt:
        case BBoxFormat.COCO2VOC  | BBoxFormat.XYWH2XYXY:
            return coco_to_voc(bbox, imgsz)
        case BBoxFormat.COCO2YOLO | BBoxFormat.XYWH2CXCYWHN:
            return coco_to_yolo(bbox, imgsz)
        case BBoxFormat.VOC2COCO  | BBoxFormat.XYXY2XYWH:
            return voc_to_coco(bbox, imgsz)
        case BBoxFormat.VOC2YOLO  | BBoxFormat.XYXY2CXCYWHN:
            return voc_to_yolo(bbox, imgsz)
        case BBoxFormat.YOLO2VOC  | BBoxFormat.CXCYWHN2XYXY:
            return yolo_to_voc(bbox, imgsz)
        case BBoxFormat.YOLO2COCO | BBoxFormat.CXCYWHN2XYXY:
            return yolo_to_coco(bbox, imgsz)
        case _:
            raise ValueError(f"``fmt`` must be one of {BBoxFormat.conversion_codes()}, got {fmt}.")
