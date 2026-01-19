#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Bounding box atomic operations.

This module provides atomic operations for bounding boxes.
"""

from __future__ import annotations

__all__ = [
    "area",
    "center",
    "center_distance",
    "ciou",
    "convert",
    "corners",
    "corners_pts",
    "crop_center",
    "crop_fit_square",
    "cxcywhn_to_xywh",
    "cxcywhn_to_xyxy",
    "denormalize",
    "diou",
    "enclosing",
    "filter_iou",
    "giou",
    "iou",
    "iou_matrix",
    "is_cxcywhn",
    "is_normalized",
    "is_xywh",
    "is_xyxy",
    "normalize",
    "pad_square",
    "split",
    "to_2d",
    "xywh_to_cxcywhn",
    "xywh_to_xyxy",
    "xyxy_to_cxcywhn",
    "xyxy_to_xywh",
]

import math
from typing import Union

import cv2
import numpy as np

from mon.core.dtypes import image as I
from mon.core.enum import BBoxFormat


# ==============================================================================
# region CREATION
# ==============================================================================


# endregion


# ==============================================================================
# region VALIDATION
# ==============================================================================

def is_normalized(bbox: np.ndarray) -> bool:
    """Check if bounding boxes are normalized.

    Args:
        bbox: Batch of bounding boxes, formatted as a numpy.ndarray of shape
            (N, 7+) and in CXCYWHN format.

    Returns:
        True if the first four values of each bounding box range from 0.0 to 1.0.
    """
    # Ensure 2D for consistent slicing
    bbox = to_2d(bbox)

    # Check if coords are in [0, 1] with a small epsilon for float precision
    return np.all(bbox[:, :4] >= -1e-5) and np.all(bbox[:, :4] <= 1.00001)


def is_cxcywhn(bbox: np.ndarray) -> bool:
    """Check if bounding boxes are in CXCYWHN format.

    Args:
        bbox: Batch of bounding boxes, formatted as a numpy.ndarray of shape
            (N, 7+).

    Returns:
        True if coordinates match CXCYWHN semantics.
    """
    # Ensure 2D for consistent slicing
    bbox = to_2d(bbox)

    if bbox.shape[0] == 0:
        return False

    # CXCYWHN is essentially any valid normalized box where W/H are positive
    return is_normalized(bbox) and np.all(bbox[:, 2:4] > 0)


def is_xyxy(bbox: np.ndarray) -> bool:
    """Check if bounding boxes are in XYXY format.

    Args:
        bbox: Batch of bounding boxes, formatted as a numpy.ndarray of shape
            (N, 7+).

    Returns:
        True if coordinates match XYXY semantics.
    """
    # Ensure 2D for consistent slicing
    bbox = to_2d(bbox)

    if bbox.shape[0] == 0:
        return False

    if is_normalized(bbox):
        return False

    # Check all boxes: x_max > x_min and y_max > y_min
    # This is the defining characteristic of XYXY
    return np.all(bbox[:, 2] > bbox[:, 0]) and np.all(bbox[:, 3] > bbox[:, 1])


def is_xywh(bbox: np.ndarray, imgsz: tuple[int, int]) -> bool:
    """Check if bounding boxes are in XYWH format.

    Args:
        bbox: Batch of bounding boxes, formatted as a numpy.ndarray of shape
            (N, 7+).
        imgsz: Image size as (H, W).

    Returns:
        True if coordinates match XYWH semantics.
    """
    # Ensure 2D for consistent slicing
    bbox = to_2d(bbox)

    if bbox.shape[0] == 0:
        return False

    if is_normalized(bbox):
        return False

    h, w = I.imgsz(imgsz)
    x1, y1, val2, val3 = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]

    # Rule 1: Width and Height must be positive
    # In XYWH, val2 and val3 are w and h.
    is_positive_dims = np.all(val2 > 0) and np.all(val3 > 0)

    # Rule 2: In XYXY, val2 (x2) must be greater than x1.
    # If there are cases where val2 < x1, it's definitely NOT XYXY.
    is_not_xyxy = np.any(val2 < x1) or np.any(val3 < y1)

    # Rule 3: Magnitude check (Heuristic)
    # If we assume XYXY, and x2 + y2 are significantly larger than width/height
    # would be in this context, we lean toward XYXY.
    # Here we check if the values at index 2,3 are "too small" to be
    # absolute coordinates for objects located at x1, y1.
    is_likely_wh = np.mean(val2) < np.mean(x1) if np.mean(x1) > w/4 else True

    return is_positive_dims and (is_not_xyxy or is_likely_wh)

# endregion


# ==============================================================================
# region RETRIEVAL
# ==============================================================================

# --- Accessing ---


# --- Selection ---

def filter_iou(bbox: np.ndarray, iou_thres: float = 0.5) -> np.ndarray:
    """Filter bounding boxes by IoU threshold using a simple area comparison.

    TODO: This is similar to Non-Maximum Suppression (NMS), but using area as the tie-breaker. Consider making a full family of NMS functions.

    Args:
        bbox: Batch of bounding boxes, formatted as a numpy.ndarray of shape
            (N, 7+) and in XYXY format.
        iou_thres: Threshold to suppress overlapping boxes. Defaults to 0.5.

    Returns:
        Filtered boxes.
    """
    if len(bbox) <= 1:
        return bbox

    # Calculate the IoU matrix once
    matrix = iou_matrix(bbox)

    # Pre-calculate areas
    areas = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])

    # Sort indices by area descending (largest boxes first)
    # This ensures we compare the 'best' candidates first
    order = areas.argsort()[::-1]
    keep  = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        # Calculate overlap of the current largest box with the rest
        # We use the pre-computed matrix for speed
        ious = matrix[i, order[1:]]

        # Identify indices of boxes that don't overlap significantly
        inds = np.where(ious < iou_thres)[0]

        # Only keep the boxes that are 'far' from the current one
        order = order[inds + 1]

    return bbox[keep]


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

def center(bbox: np.ndarray) -> np.ndarray:
    """Calculate the center point(s) of bounding box(es).

    Args:
        bbox: Batch of bounding boxes, formatted as a numpy.ndarray of shape
            (N, 7+) and in XYXY format.

    Returns:
        Array of center points with shape (N, 2).
    """
    # Ensure 2D for consistent slicing
    bbox = to_2d(bbox)
    # Optimized: (x1 + x2) / 2 is mathematically identical and slightly faster
    # than x1 + (x2 - x1) / 2
    return (bbox[:, :2] + bbox[:, 2:4]) / 2.0


def corners(bbox: np.ndarray) -> np.ndarray:
    """Get corner coordinates for bounding boxes.

    Args:
        bbox: Batch of bounding boxes, formatted as a numpy.ndarray of shape
            (N, 7+) and in XYXY format.

    Returns:
        Array of corner coordinates with shape (N, 8), with each corner in
        [x1, y1, x2, y1, x2, y2, x1, y2] format.
    """
    # Ensure 2D for consistent slicing
    bbox = to_2d(bbox)
    x1, y1, x2, y2 = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
    # Standard order: Top-Left, Top-Right, Bottom-Right, Bottom-Left
    return np.stack([x1, y1, x2, y1, x2, y2, x1, y2], axis=-1)


def corners_pts(bbox: np.ndarray) -> np.ndarray:
    """Get corner coordinates for bounding boxes.

    Args:
        bbox: Batch of bounding boxes, formatted as a numpy.ndarray of shape
            (N, 7+) and in XYXY format.

    Returns:
        Array of corner points of shape (N, 4, 2), with each corner point is
        a pair of (x, y) coordinates.
    """
    c = corners(bbox)
    return c.reshape(-1, 4, 2)


def enclosing(bbox: np.ndarray) -> np.ndarray:
    """Get the enclosing XYXY boxes for corner-format boxes.

    Args:
        bbox: Corner-format boxes with last dim >= 8.

    Returns:
        Enclosing XYXY boxes with any extra fields preserved.

    Raises:
        ValueError: If ``bbox``'s the last dimension < 8.
    """
    if bbox.shape[-1] < 8:
        raise ValueError(f"Expected corner format (last dim >= 8), but got {bbox.shape[-1]}.")

    # Efficiently separate X and Y coordinates
    x_coords = bbox[:, 0:8:2] # Slicing [start:stop:step]
    y_coords = bbox[:, 1:8:2]

    x1 = np.min(x_coords, axis=1)
    y1 = np.min(y_coords, axis=1)
    x2 = np.max(x_coords, axis=1)
    y2 = np.max(y_coords, axis=1)

    # Reassemble with remaining attributes (conf, class, id)
    return np.column_stack([x1, y1, x2, y2, bbox[:, 8:]])


def area(bbox: np.ndarray) -> np.ndarray:
    """Calculate the area(s) of bounding box(es).

    Args:
        bbox: Batch of bounding boxes, formatted as a numpy.ndarray of shape
            (N, 7+) and in XYXY format.

    Returns:
        Array of areas with shape (N, ).
    """
    # Ensure 2D for consistent slicing
    bbox = to_2d(bbox)
    # Use clip to ensure area is never negative
    return np.maximum(0, bbox[:, 2] - bbox[:, 0]) * np.maximum(0, bbox[:, 3] - bbox[:, 1])


def center_distance(
    bbox1: np.ndarray,
    bbox2: np.ndarray,
    imgsz: tuple[int, int] = None
) -> np.ndarray:
    """Compute normalized inverted center distances between box sets.

    Args:
        bbox1: Batch of bounding boxes, formatted as a numpy.ndarray of shape
            (N, 7+) and in XYXY format.
        bbox2: Batch of bounding boxes, formatted as a numpy.ndarray of shape
            (M, 7+) and in XYXY format.
        imgsz: Image size as (H, W). If provided, normalization is done by
            the image diagonal. If None, normalization is done by the maximum
            distance in the current pair. Defaults to None.

    Returns:
        Normalized inverted distance, formatted as a numpy.ndarray of shape
        (N, M) and values ranging from 0.0 to 1.0, where a smaller geometric
        distance yields a larger value.
    """
    # Ensure 2D arrays
    bbox1 = to_2d(bbox1)
    bbox2 = to_2d(bbox2)

    # Calculate centers: (N, 2) and (M, 2)
    c1 = (bbox1[:, :2] + bbox1[:, 2:4]) / 2.0
    c2 = (bbox2[:, :2] + bbox2[:, 2:4]) / 2.0

    # Broadcast to (N, M, 2)
    # Using None indexing is slightly cleaner than expand_dims for pairwise ops
    dist = np.linalg.norm(c1[:, None, :] - c2[None, :, :], axis=-1)

    # Normalize
    if imgsz is not None:
        # Normalize by the image diagonal (constant across frames)
        max_dist = np.sqrt(imgsz[0]**2 + imgsz[1]**2)
    else:
        # Normalize by the largest distance in the current pair
        max_dist = np.max(dist)

    if max_dist > 1e-7:
        # Invert: 1.0 is closest, 0.0 is furthest
        return 1.0 - (dist / max_dist)

    return np.ones_like(dist)


def iou(bbox1: np.ndarray, bbox2: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Compute pairwise Intersection over Union (IoU) between two sets of
    bounding boxes.

    Args:
        bbox1: Batch of bounding boxes, formatted as a numpy.ndarray of shape
            (N, 7+) and in XYXY format.
        bbox2: Batch of bounding boxes, formatted as a numpy.ndarray of shape
            (M, 7+) and in XYXY format.
        eps: Small value to prevent division by zero. Defaults to 1e-7.

    Returns:
        IoU matrix, formatted as a numpy.ndarray of shape (N, M).
    """
    # Ensure 2D arrays
    bbox1 = to_2d(bbox1)
    bbox2 = to_2d(bbox2)

    # Intersection coordinates
    # bbox1[:, None, :2] is shape (N, 1, 2)
    # bbox2[None, :, :2] is shape (1, M, 2)
    lt = np.maximum(bbox1[:, None, :2], bbox2[None, :, :2])  # left-top
    rb = np.minimum(bbox1[:, None, 2:4], bbox2[None, :, 2:4]) # right-bottom

    # Intersection Area
    wh    = np.maximum(0.0, rb - lt)
    inter = wh[:, :, 0] * wh[:, :, 1]

    # Individual Areas
    area1 = (bbox1[:, 2] - bbox1[:, 0]) * (bbox1[:, 3] - bbox1[:, 1])
    area2 = (bbox2[:, 2] - bbox2[:, 0]) * (bbox2[:, 3] - bbox2[:, 1])

    # Union Area: Area1 + Area2 - Intersection
    # area1[:, None] is (N, 1), area2[None, :] is (1, M)
    union = area1[:, None] + area2[None, :] - inter

    return inter / (union + eps)


def giou(bbox1: np.ndarray, bbox2: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Compute generalized IoU (GIoU) between two sets of bounding boxes.

    Args:
        bbox1: Batch of bounding boxes, formatted as a numpy.ndarray of shape
            (N, 7+) and in XYXY format.
        bbox2: Batch of bounding boxes, formatted as a numpy.ndarray of shape
            (M, 7+) and in XYXY format.
        eps: Small value to prevent division by zero. Defaults to 1e-7.

    Returns:
        GIoU matrix, formatted as a numpy.ndarray of shape (N, M).
    """
    # Ensure 2D arrays
    bbox1 = to_2d(bbox1)
    bbox2 = to_2d(bbox2)

    # Standard IoU Calculation
    # lt: left-top, rb: right-bottom
    lt    = np.maximum(bbox1[:, None, :2], bbox2[None, :, :2])
    rb    = np.minimum(bbox1[:, None, 2:4], bbox2[None, :, 2:4])
    wh    = np.maximum(0.0, rb - lt)
    inter = wh[..., 0] * wh[..., 1]

    area1 = (bbox1[:, 2] - bbox1[:, 0]) * (bbox1[:, 3] - bbox1[:, 1])
    area2 = (bbox2[:, 2] - bbox2[:, 0]) * (bbox2[:, 3] - bbox2[:, 1])
    union = area1[:, None] + area2[None, :] - inter + eps
    iou_  = inter / union

    # Enclosing Box (C)
    # Finding the min of the mins and max of the maxes
    c_lt   = np.minimum(bbox1[:, None, :2], bbox2[None, :, :2])
    c_rb   = np.maximum(bbox1[:, None, 2:4], bbox2[None, :, 2:4])
    c_wh   = np.maximum(0.0, c_rb - c_lt)
    area_c = c_wh[..., 0] * c_wh[..., 1] + eps

    # GIoU formula
    return iou_ - (area_c - union) / area_c


def diou(bbox1: np.ndarray, bbox2: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Compute distance IoU (DIoU) between two sets of bounding boxes.

    Args:
        bbox1: Batch of bounding boxes, formatted as a numpy.ndarray of shape
            (N, 7+) and in XYXY format.
        bbox2: Batch of bounding boxes, formatted as a numpy.ndarray of shape
            (M, 7+) and in XYXY format.
        eps: Small value to prevent division by zero. Defaults to 1e-7.

    Returns:
        DIoU matrix, formatted as a numpy.ndarray of shape (N, M).
    """
    # Ensure 2D arrays
    bbox1 = to_2d(bbox1)
    bbox2 = to_2d(bbox2)

    # Standard IoU components
    # Using [:, None, :] and [None, :, :] is a faster alternative to expand_dims
    b1_lt, b1_rb = bbox1[:, :2], bbox1[:, 2:4]
    b2_lt, b2_rb = bbox2[:, :2], bbox2[:, 2:4]

    # Intersection
    inter_lt   = np.maximum(b1_lt[:, None, :], b2_lt[None, :, :])
    inter_rb   = np.minimum(b1_rb[:, None, :], b2_rb[None, :, :])
    inter_wh   = np.maximum(0.0, inter_rb - inter_lt)
    inter_area = inter_wh[..., 0] * inter_wh[..., 1]

    # Union
    area1 = (b1_rb[:, 0] - b1_lt[:, 0]) * (b1_rb[:, 1] - b1_lt[:, 1])
    area2 = (b2_rb[:, 0] - b2_lt[:, 0]) * (b2_rb[:, 1] - b2_lt[:, 1])
    union = area1[:, None] + area2[None, :] - inter_area + eps
    iou_  = inter_area / union

    # DIoU Penalty Term
    # Center points
    c1   = (b1_lt + b1_rb) / 2.0
    c2   = (b2_lt + b2_rb) / 2.0
    # Squared Euclidean distance between centers
    rho2 = np.sum((c1[:, None, :] - c2[None, :, :]) ** 2, axis=-1)

    # Smallest enclosing box diagonal
    enc_lt  = np.minimum(b1_lt[:, None, :], b2_lt[None, :, :])
    enc_rb  = np.maximum(b1_rb[:, None, :], b2_rb[None, :, :])
    enc_wh  = enc_rb - enc_lt
    c2_diag = np.sum(enc_wh ** 2, axis=-1) + eps  # Squared diagonal length

    return iou_ - (rho2 / c2_diag)


def ciou(bbox1: np.ndarray, bbox2: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Compute complete IoU (CIoU) between two sets of bounding boxes.

    Args:
        bbox1: Batch of bounding boxes, formatted as a numpy.ndarray of shape
            (N, 7+) and in XYXY format.
        bbox2: Batch of bounding boxes, formatted as a numpy.ndarray of shape
            (M, 7+) and in XYXY format.
        eps: Small value to prevent division by zero. Defaults to 1e-7.

    Returns:
        CIoU matrix, formatted as a numpy.ndarray of shape (N, M).
    """
    # Ensure 2D arrays
    bbox1 = to_2d(bbox1)
    bbox2 = to_2d(bbox2)

    # Broadcast to (N, M, 4)
    b1 = bbox1[:, None, :]
    b2 = bbox2[None, :, :]

    # IoU Calculation
    inter_lt   = np.maximum(b1[..., :2], b2[..., :2])
    inter_rb   = np.minimum(b1[..., 2:4], b2[..., 2:4])
    inter_wh   = np.maximum(0.0, inter_rb - inter_lt)
    inter_area = inter_wh[..., 0] * inter_wh[..., 1]

    area1 = (b1[..., 2] - b1[..., 0]) * (b1[..., 3] - b1[..., 1])
    area2 = (b2[..., 2] - b2[..., 0]) * (b2[..., 3] - b2[..., 1])
    union = area1 + area2 - inter_area + eps
    iou_ = inter_area / union

    # Distance Term (DIoU)
    c1   = (b1[..., :2] + b1[..., 2:4]) / 2.0
    c2   = (b2[..., :2] + b2[..., 2:4]) / 2.0
    rho2 = np.sum((c1 - c2) ** 2, axis=-1)

    enc_lt  = np.minimum(b1[..., :2], b2[..., :2])
    enc_rb  = np.maximum(b1[..., 2:4], b2[..., 2:4])
    enc_wh  = np.maximum(0.0, enc_rb - enc_lt)
    c2_diag = np.sum(enc_wh ** 2, axis=-1) + eps

    # Aspect Ratio Term (CIoU)
    w1, h1 = (b1[..., 2] - b1[..., 0]), (b1[..., 3] - b1[..., 1])
    w2, h2 = (b2[..., 2] - b2[..., 0]), (b2[..., 3] - b2[..., 1])

    # Standard CIoU v calculation
    v = (4 / np.pi**2) * np.power(np.arctan(w2 / (h2 + eps)) - np.arctan(w1 / (h1 + eps)), 2)

    # alpha is defined such that it prioritizes the overlap over aspect ratio
    with np.errstate(divide='ignore', invalid='ignore'):
        alpha = v / (v - iou_ + 1.0 + eps)

    return iou_ - (rho2 / c2_diag) - (alpha * v)


def iou_matrix(bbox: np.ndarray, self_match: bool = False) -> np.ndarray:
    """Compute pairwise IoU matrix between all bounding boxes in a set.

    Args:
        bbox: Batch of bounding boxes, formatted as a numpy.ndarray of shape
            (N, 7+) and in XYXY format.
        self_match: If True, include self-comparisons (diagonal will be 1s).
            Defaults to False.

    Returns:
        IoU matrix, formatted as a numpy.ndarray of shape (N, N).
    """
    # Ensure 2D arrays
    bbox = to_2d(bbox)

    # Slice once to avoid repeated indexing
    x1, y1, x2, y2 = bbox[:, 0:1], bbox[:, 1:2], bbox[:, 2:3], bbox[:, 3:4]

    # Vectorized Intersection
    inter = (np.maximum(0, np.minimum(x2, x2.T) - np.maximum(x1, x1.T)) *
             np.maximum(0, np.minimum(y2, y2.T) - np.maximum(y1, y1.T)))

    # Vectorized Union
    area_ = (x2 - x1) * (y2 - y1)
    union = area_ + area_.T - inter

    # eps = 1e-7 to prevent division by zero
    iou_mat = np.divide(inter, union, out=np.zeros_like(inter), where=union > 1e-7)

    if not self_match:
        np.fill_diagonal(iou_mat, 0)

    return iou_mat

# endregion


# ==============================================================================
# region TRANSFORMATION
# ==============================================================================

# --- Casting ---

def xywh_to_cxcywhn(bbox: np.ndarray, imgsz: tuple[int, int]) -> np.ndarray:
    """Convert bounding boxes from XYWH to CXCYWHN format.

    Args:
        bbox: Batch of bounding boxes, formatted as a numpy.ndarray of shape
            (N, 7+) and in XYWH format.
        imgsz: Image size as (H, W).

    Returns:
        Bounding boxes in CXCYWHN format.
    """
    # Standardize image size
    # Ensure w0 and h0 are floats to prevent integer division issues
    imgsz  = I.imgsz(imgsz)
    h0, w0 = float(imgsz[0]), float(imgsz[1])

    # Ensure 2D for consistent slicing
    bbox = to_2d(bbox)

    # Handle empty inputs
    if bbox.shape[0] == 0:
        return bbox.copy()

    # Vectorized transformation
    # We slice up to 4 to isolate coordinates, then grab the rest
    coords = bbox[:, :4]
    rest   = bbox[:, 4:]

    x, y, w, h = coords.T

    cx_n = (x + w / 2.0) / w0
    cy_n = (y + h / 2.0) / h0
    w_n  = w / w0
    h_n  = h / h0

    # Reassemble
    return np.column_stack((cx_n, cy_n, w_n, h_n, rest))


def xywh_to_xyxy(bbox: np.ndarray, imgsz: tuple[int, int]) -> np.ndarray:
    """Convert bounding boxes from XYWH to XYXY format.

    Args:
        bbox: Batch of bounding boxes, formatted as a numpy.ndarray of shape
            (N, 7+) and in XYWH format.
        imgsz: Image size as (H, W).

    Returns:
        Bounding boxes in XYXY format.
    """
    # Ensure 2D for consistent slicing
    bbox = to_2d(bbox)

    # Handle empty inputs
    if bbox.shape[0] == 0:
        return bbox.copy()

    # Vectorized transformation
    # We slice up to 4 to isolate coordinates, then grab the rest
    coords = bbox[:, :4]
    rest   = bbox[:, 4:]

    x, y, w, h = coords.T

    # Calculate new boundaries
    x2 = x + w
    y2 = y + h

    # Reassemble
    return np.column_stack((x, y, x2, y2, rest))


def xyxy_to_cxcywhn(bbox: np.ndarray, imgsz: tuple[int, int]) -> np.ndarray:
    """Convert bounding boxes from XYXY to normalized CXCYWHN.

    Args:
        bbox: Batch of bounding boxes, formatted as a numpy.ndarray of shape
            (N, 7+) and in XYXY format.
        imgsz: Image size as (H, W).

    Returns:
        Bounding boxes in CXCYWHN format.
    """
    # Standardize image size
    # Ensure w0 and h0 are floats to prevent integer division issues
    imgsz  = I.imgsz(imgsz)
    h0, w0 = float(imgsz[0]), float(imgsz[1])

    # Ensure 2D for consistent slicing
    bbox = to_2d(bbox)

    # Handle empty inputs
    if bbox.shape[0] == 0:
        return bbox.copy()

    # Vectorized transformation
    # We slice up to 4 to isolate coordinates, then grab the rest
    coords = bbox[:, :4]
    rest   = bbox[:, 4:]

    x1, y1, x2, y2 = coords.T

    # Compute normalized CXCYWH
    # Use 1e-7 to prevent division by zero if imgsz is invalid
    eps  = 1e-7
    w    = x2 - x1
    h    = y2 - y1

    cx_n = (x1 + w / 2.0) / (w0 + eps)
    cy_n = (y1 + h / 2.0) / (h0 + eps)
    w_n  = w / (w0 + eps)
    h_n  = h / (h0 + eps)

    # Reassemble
    return np.column_stack((cx_n, cy_n, w_n, h_n, rest))


def xyxy_to_xywh(bbox: np.ndarray, imgsz: tuple[int, int]) -> np.ndarray:
    """Convert bounding boxes from XYXY to XYWH format.

    Args:
        bbox: Batch of bounding boxes, formatted as a numpy.ndarray of shape
            (N, 7+) and in XYXY format.
        imgsz: Image size as (H, W).

    Returns:
        Bounding boxes in XYWH format.
    """
    # Ensure 2D for consistent slicing
    bbox = to_2d(bbox)

    # Handle empty inputs
    if bbox.shape[0] == 0:
        return bbox.copy()

    # Vectorized transformation
    # We slice up to 4 to isolate coordinates, then grab the rest
    coords = bbox[:, :4]
    rest   = bbox[:, 4:]

    x1, y1, x2, y2 = coords.T

    # Compute width and height
    w = x2 - x1
    h = y2 - y1

    # Reassemble
    return np.column_stack((x1, y1, w, h, rest))


def cxcywhn_to_xywh(bbox: np.ndarray, imgsz: tuple[int, int]) -> np.ndarray:
    """Convert bounding boxes from CXCYWHN to XYWH.

    Args:
        bbox: Batch of bounding boxes, formatted as a numpy.ndarray of shape
            (N, 7+) and in CXCYWHN format.
        imgsz: Image size as (H, W).

    Returns:
        Bounding boxes in XYWH format.
    """
    # Standardize image size
    # Ensure w0 and h0 are floats to prevent integer division issues
    imgsz  = I.imgsz(imgsz)
    h0, w0 = float(imgsz[0]), float(imgsz[1])

    # Ensure 2D for consistent slicing
    bbox = to_2d(bbox)

    # Handle empty inputs
    if bbox.shape[0] == 0:
        return bbox.copy()

    # Vectorized transformation
    # We slice up to 4 to isolate coordinates, then grab the rest
    coords = bbox[:, :4]
    rest   = bbox[:, 4:]

    cx_n, cy_n, w_n, h_n = coords.T

    # Denormalize and Shift
    w = w_n * w0
    h = h_n * h0
    x = (cx_n * w0) - (w / 2.0)
    y = (cy_n * h0) - (h / 2.0)

    # Reassemble
    return np.column_stack((x, y, w, h, rest))


def cxcywhn_to_xyxy(bbox: np.ndarray, imgsz: tuple[int, int]) -> np.ndarray:
    """Convert bounding boxes from CXCYWHN to XYXY.

    Args:
        bbox: Batch of bounding boxes, formatted as a numpy.ndarray of shape
            (N, 7+) and in CXCYWHN format.
        imgsz: Image size as (H, W).

    Returns:
        Bounding boxes in XYXY format.
    """
    # Standardize image size
    # Ensure w0 and h0 are floats to prevent integer division issues
    imgsz  = I.imgsz(imgsz)
    h0, w0 = float(imgsz[0]), float(imgsz[1])

    # Ensure 2D for consistent slicing
    bbox = to_2d(bbox)

    # Handle empty inputs
    if bbox.shape[0] == 0:
        return bbox.copy()

    # Vectorized transformation
    # We slice up to 4 to isolate coordinates, then grab the rest
    coords = bbox[:, :4]
    rest   = bbox[:, 4:]

    cx_n, cy_n, w_n, h_n = coords.T

    # Transform to absolute corners
    # Applying the width/height offset before multiplying by imgsz
    # is mathematically identical but slightly cleaner.
    x1 = (cx_n - w_n / 2) * w0
    y1 = (cy_n - h_n / 2) * h0
    x2 = (cx_n + w_n / 2) * w0
    y2 = (cy_n + h_n / 2) * h0

    # Reassemble
    return np.column_stack((x1, y1, x2, y2, rest))


def convert(bbox: np.ndarray, fmt: BBoxFormat, imgsz: tuple[int, int]) -> np.ndarray:
    """Convert bounding boxes between supported formats.

    Args:
        bbox: Batch of bounding boxes, formatted as a numpy.ndarray of shape
            (N, 7+).
        fmt: Target conversion format.
        imgsz: Image size as (H, W).

    Returns:
        Bounding boxes in the desired format.

    Raises:
        ValueError: If ``fmt`` is invalid or unsupported.
    """
    if len(bbox) == 0:
        return bbox

    # Normalize the enum input
    if not isinstance(fmt, BBoxFormat):
        fmt = BBoxFormat(fmt)

    match fmt:
        # COCO (XYWH) -> Target
        case BBoxFormat.COCO2VOC | BBoxFormat.XYWH2XYXY:
            return xywh_to_xyxy(bbox, imgsz)
        case BBoxFormat.COCO2YOLO | BBoxFormat.XYWH2CXCYWHN:
            return xywh_to_cxcywhn(bbox, imgsz)

        # VOC (XYXY) -> Target
        case BBoxFormat.VOC2COCO | BBoxFormat.XYXY2XYWH:
            return xyxy_to_xywh(bbox, imgsz)
        case BBoxFormat.VOC2YOLO | BBoxFormat.XYXY2CXCYWHN:
            return xyxy_to_cxcywhn(bbox, imgsz)

        # YOLO (CXCYWHN) -> Target
        case BBoxFormat.YOLO2VOC | BBoxFormat.CXCYWHN2XYXY:
            return cxcywhn_to_xyxy(bbox, imgsz)
        case BBoxFormat.YOLO2COCO | BBoxFormat.CXCYWHN2XYWH:
            return cxcywhn_to_xywh(bbox, imgsz)

        case _:
            raise ValueError(f"Unsupported 'fmt' conversion: {fmt}. "
                             f"Must be one of: {BBoxFormat.conversion_codes()}.")


# --- Encoding ---


# --- Standardization ---


# --- Structural ---

def to_2d(bbox: Union[np.ndarray, list, tuple]) -> np.ndarray:
    """Convert bounding boxes to a 2-D numpy.ndarray.

    Args:
        bbox: Single or a batch of bounding boxes.

    Returns:
        A numpy.ndarray of dimensions (N, M).

    Raises:
        ValueError: If ``bbox``'s type is unsupported.
        TypeError: If list/tuple elements have inconsistent shapes.
    """
    # Convert lists/tuples to array immediately for standardized processing
    if isinstance(bbox, (list, tuple)):
        try:
            bbox = np.array(bbox, dtype=np.float32)
        except ValueError:
            # Handle jagged arrays (e.g., one box has 7 elements, another has 8)
            raise ValueError("Expected all elements in 'bbox' to have the same shape.")

    if not isinstance(bbox, np.ndarray):
        raise TypeError(f"Expected 'bbox' to be a numpy.ndarray, list, or tuple, "
                        f"but got {type(bbox).__name__}.")

    # Handle various NumPy shapes
    if bbox.ndim == 1:
        return bbox[np.newaxis, :]   # [5+] -> [1, 5+]
    elif bbox.ndim == 3:
        return np.squeeze(bbox)      # [1, N, 5+] -> [N, 5+]

    return bbox


def split(image: np.ndarray, bbox : np.ndarray, n: int = 2) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Split an image into ``n`` tiles and adjust bounding boxes per tile.

    Args:
        image: RGB or grayscale image, formatted as a numpy.ndarray of shape
            (H, W, C) and pixel values ranging from 0 to 255.
        bbox: Batch of bounding boxes, formatted as a numpy.ndarray of shape
            (N, 7+) and in CXCYWHN format.
        n: Number of tiles to split into. Defaults to 2.

    Returns:
        Tuple of two lists: sub-images and their corresponding bounding boxes.

    Raises:
        ValueError: If ``image`` is not a numpy.ndarray of shape (H, W, C).
        ValueError: If ``bbox`` is not a numpy.ndarray of shape (N, 7+).
        ValueError: If ``n`` is less than 1 or exceeds the number of pixels in
            the original image.
    """
    if not isinstance(image, np.ndarray) or len(image.shape) != 3:
        raise ValueError(f"Expected 'image' to be a numpy.ndarray of shape (H, W, C), "
                         f"but got {image.shape}.")
    if not isinstance(bbox, np.ndarray) or bbox.ndim != 2:
        raise ValueError(f"Expected 'bbox' to be a numpy.ndarray of shape (N, 7+), "
                         f"but got {bbox.shape}.")
    if n < 1:
        raise ValueError(f"Expected 'n' to be >= 1, but got {n}.")

    h0, w0 = I.imgsz(image)
    if n > h0 * w0:
        raise ValueError(f"Expected 'n' to be <= {h0 * w0}, but got {n}.")

    # Determine orientation
    is_portrait = h0 > w0

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
    sub_h      = h0 // rows
    sub_w      = w0 // cols
    sub_images = []
    sub_bboxes = []

    # Pre-convert all bboxes to global XYXY once to avoid repeated math
    bbox_xyxy_global = convert(bbox, fmt=BBoxFormat.CXCYWHN2XYXY, imgsz=(h0, w0))

    for i in range(rows):
        for j in range(cols):
            if len(sub_images) >= n:
                break

            y_start, x_start = i * sub_h, j * sub_w
            y_end, x_end     = min(y_start + sub_h, h0), min(x_start + sub_w, w0)

            tile   = image[y_start:y_end, x_start:x_end].copy()
            th, tw = tile.shape[:2]  # Actual tile dimensions
            sub_images.append(tile)

            if len(bbox) == 0:
                sub_bboxes.append(np.zeros((0, bbox.shape[1]), dtype=np.float32))
                continue

            # Vectorized adjustment for this specific tile
            tile_bboxes = bbox_xyxy_global.copy()
            tile_bboxes[:, [0, 2]] -= x_start
            tile_bboxes[:, [1, 3]] -= y_start

            # Clip to tile boundaries
            tile_bboxes[:, [0, 2]] = np.clip(tile_bboxes[:, [0, 2]], 0, tw)
            tile_bboxes[:, [1, 3]] = np.clip(tile_bboxes[:, [1, 3]], 0, th)

            # Filter out boxes that have no area in this tile
            keep = (tile_bboxes[:, 2] > tile_bboxes[:, 0]) & (tile_bboxes[:, 3] > tile_bboxes[:, 1])
            valid_bboxes = tile_bboxes[keep]

            if len(valid_bboxes) > 0:
                # Re-normalize to the TILE'S dimensions
                norm_bboxes = convert(valid_bboxes, fmt=BBoxFormat.XYXY2CXCYWHN, imgsz=(th, tw))
                sub_bboxes.append(norm_bboxes)
            else:
                sub_bboxes.append(np.zeros((0, bbox.shape[1]), dtype=np.float32))

    return sub_images, sub_bboxes


# --- Statistical ---

def normalize(bbox: np.ndarray, imgsz: tuple[int, int]) -> np.ndarray:
    """Normalize bounding boxes by image size.

    Args:
        bbox: Batch of bounding boxes, formatted as a numpy.ndarray of shape
            (N, 7+).
        imgsz: Image size as (H, W).

    Returns:
        Normalized bounding boxes, formatted as a numpy.ndarray of
        dimensions (N, 7+) and values ranging from 0 to 1.
    """
    # Standardize image size
    # Ensure w0 and h0 are floats to prevent integer division issues
    imgsz  = I.imgsz(imgsz)
    h0, w0 = float(imgsz[0]), float(imgsz[1])

    # Ensure 2D for consistent slicing
    bbox = to_2d(bbox)

    if is_normalized(bbox):
        return bbox

    # Vectorized approach
    # We create a normalization vector [W, H, W, H] to divide the first 4 columns
    # This is faster than unpacking/stacking for large N
    norm_vec = np.array([w0, h0, w0, h0], dtype=np.float32)

    # Clone to avoid modifying the original array in-place
    normalized_bbox         = bbox.astype(np.float32).copy()
    normalized_bbox[:, :4] /= (norm_vec + 1e-7) # Add epsilon for stability

    return normalized_bbox


def denormalize(bbox: np.ndarray, imgsz: tuple[int, int]) -> np.ndarray:
    """Denormalize bounding boxes to pixel units.

    Args:
        bbox: Batch of bounding boxes, formatted as a numpy.ndarray of shape
            (N, 7+).
        imgsz: Image size as (H, W).

    Returns:
        Denormalized bounding boxes, formatted as a numpy.ndarray of shape
            (N, 7+) and values ranging from 0 to 255.
    """
    # Standardize image size
    # Ensure w0 and h0 are floats to prevent integer division issues
    imgsz  = I.imgsz(imgsz)
    h0, w0 = float(imgsz[0]), float(imgsz[1])

    # Ensure 2D for consistent slicing
    bbox = to_2d(bbox)

    # Skip if already in pixel units or empty
    if bbox.shape[0] == 0 or not is_normalized(bbox):
        return bbox

    # Create scaling vector [W, H, W, H]
    # This works for both XYXY and CXCYWH formats
    scale_vec = np.array([w0, h0, w0, h0], dtype=np.float32)

    # Apply scaling to the first 4 columns only
    denormalized_bbox         = bbox.copy()
    denormalized_bbox[:, :4] *= scale_vec

    return denormalized_bbox


# --- Geometric ---

def crop_center(
    image: np.ndarray,
    bbox : np.ndarray,
    imgsz: int
) -> tuple[np.ndarray, np.ndarray]:
    """Center-crop an image and adjust accompanying bounding boxes.

    Args:
        image: RGB or grayscale image, formatted as a numpy.ndarray of shape
            (H, W, C) and pixel values ranging from 0 to 255.
        bbox: Batch of bounding boxes, formatted as a numpy.ndarray of shape
            (N, 7+) and in CXCYWHN format.
        imgsz: Target image size as (H, W) for cropping.

    Returns:
        Tuple of cropped image and adjusted bounding boxes.

    Raises:
        ValueError: If ``imgsz`` exceeds the original ``image``'s size.
    """
    h0, w0 = I.imgsz(image)
    h1, w1 = I.imgsz(imgsz)
    if h1 > h0 or w1 > w0:
        raise ValueError(f"Target 'imgsz' {imgsz} exceeds original image size {image.shape[:2]}.")

    # Calculate crop region (center of image)
    x_start = max(0, (w0 - w1) // 2)
    y_start = max(0, (h0 - h1) // 2)
    x_end   = x_start + w1
    y_end   = y_start + h1

    # Crop Image
    cropped_image = image[y_start:y_end, x_start:x_end].copy()

    if len(bbox) == 0:
        return cropped_image, bbox

    # Convert to XYXY in absolute pixels
    bbox_xyxy = convert(bbox, fmt=BBoxFormat.CXCYWHN2XYXY, imgsz=(h0, w0))

    # Vectorized Shift
    # Shift x1, y1, x2, y2 all at once
    bbox_xyxy[:, [0, 2]] -= x_start
    bbox_xyxy[:, [1, 3]] -= y_start

    # Vectorized Clipping
    # x1, x2 clipped to [0, w1] | y1, y2 clipped to [0, h1]
    bbox_xyxy[:, [0, 2]] = np.clip(bbox_xyxy[:, [0, 2]], 0, w1)
    bbox_xyxy[:, [1, 3]] = np.clip(bbox_xyxy[:, [1, 3]], 0, h1)

    # Vectorized Filtering
    # Keep only boxes where x2 > x1 AND y2 > y1 (valid area)
    keep          = (bbox_xyxy[:, 2] > bbox_xyxy[:, 0]) & (bbox_xyxy[:, 3] > bbox_xyxy[:, 1])
    adjusted_bbox = bbox_xyxy[keep]

    # Re-normalize to target crop size
    if len(adjusted_bbox) > 0:
        adjusted_bbox = convert(adjusted_bbox, fmt=BBoxFormat.XYXY2CXCYWHN, imgsz=(h1, w1))

    return cropped_image, adjusted_bbox


def crop_fit_square(
    image    : np.ndarray,
    bbox     : np.ndarray,
    pad_value: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Crop background content and pad to a centered square.

    Args:
        image: RGB or grayscale image, formatted as a numpy.ndarray of shape
            (H, W, C) and pixel values ranging from 0 to 255.
        bbox: Batch of bounding boxes, formatted as a numpy.ndarray of shape
            (N, 7+) and in CXCYWHN format.
        pad_value: Padding value. Defaults to 0.

    Returns:
        Tuple of padded image and adjusted bounding boxes.
    """
    h0, w0 = I.imgsz(image)

    # Find non-border pixels
    if len(image.shape) == 3 and image.shape[2] == 3:
        mask = np.any(image != pad_value, axis=2)
    else:
        mask = image != pad_value

    coords = np.argwhere(mask)
    if coords.size == 0:
        # If the entire image is border color, return a square of pad_value
        dim = max(h0, w0)
        padded_image = np.full((dim, dim, image.shape[2]), pad_value, dtype=image.dtype)
        return padded_image, np.array([], np.float32).reshape(0, bbox.shape[1])

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0) + 1  # Add 1 to include the max pixel

    # Crop the image
    cropped_image = image[y_min:y_max, x_min:x_max].copy()
    h1, w1        = I.imgsz(cropped_image)

    # Pad to make square
    dim   = max(h1, w1)
    pad_h = (dim - h1) // 2
    pad_w = (dim - w1) // 2

    padded_image = cv2.copyMakeBorder(
        src        = cropped_image,
        top        = pad_h,
        bottom     = dim - h1 - pad_h,
        left       = pad_w,
        right      = dim - w1 - pad_w,
        borderType = cv2.BORDER_CONSTANT,
        value      = [pad_value, pad_value, pad_value],
    )

    # Vectorized BBox Logic
    if bbox.size > 0:
        bbox_xyxy = convert(bbox, fmt=BBoxFormat.CXCYWHN2XYXY, imgsz=(h0, w0))

        # Shift and Pad
        bbox_xyxy[:, [0, 2]] = bbox_xyxy[:, [0, 2]] - x_min + pad_w
        bbox_xyxy[:, [1, 3]] = bbox_xyxy[:, [1, 3]] - y_min + pad_h

        # Re-normalize to the square 'dim'
        adjusted_bbox = convert(bbox_xyxy, fmt=BBoxFormat.XYXY2CXCYWHN, imgsz=(dim, dim))
    else:
        adjusted_bbox = np.empty((0, bbox.shape[1]), dtype=np.float32)

    return padded_image, adjusted_bbox


def pad_square(
    image    : np.ndarray,
    bbox     : np.ndarray,
    pad_value: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Pad an image to a centered square and adjust bboxes.

    Args:
        image: RGB or grayscale image, formatted as a numpy.ndarray of shape
            (H, W, C) and pixel values ranging from 0 to 255.
        bbox: Batch of bounding boxes, formatted as a numpy.ndarray of shape
            (N, 7+) and in CXCYWHN format.
        pad_value: Padding value. Defaults to 0.

    Returns:
        Tuple of padded image and adjusted bounding boxes.
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
        value      = [pad_value] * (image.shape[2] if len(image.shape) == 3 else 1),
    )

    # Vectorized Coordinate Adjustment
    if bbox.shape[0] == 0:
        return padded_image, bbox

    # Pixel space (normalized to original size)
    bbox_xyxy = convert(bbox, fmt=BBoxFormat.CXCYWHN2XYXY, imgsz=(h0, w0))

    # Add offsets to x1, y1, x2, y2
    bbox_xyxy[:, [0, 2]] += pad_w
    bbox_xyxy[:, [1, 3]] += pad_h

    # Re-normalize to the new square size (dim)
    adjusted_bbox = convert(bbox_xyxy, fmt=BBoxFormat.XYXY2CXCYWHN, imgsz=(dim, dim))

    return padded_image, adjusted_bbox

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
