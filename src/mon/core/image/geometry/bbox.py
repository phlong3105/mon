#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Bounding Box Geometry.

This module implements geometry functions for bounding boxes. For handling
geometry, :obj:`numpy.ndarray` is used as the primary data structure.
"""

from __future__ import annotations

__all__ = [
    "bbox_area",
    "bbox_center",
    "bbox_center_distance",
    "bbox_ciou",
    "bbox_coco_to_voc",
    "bbox_coco_to_yolo",
    "bbox_corners",
    "bbox_corners_points",
    "bbox_cxcywhn_to_xywh",
    "bbox_cxcywhn_to_xyxy",
    "bbox_cxcywhn_to_xyxyn",
    "bbox_diou",
    "bbox_giou",
    "bbox_iou",
    "bbox_voc_to_coco",
    "bbox_voc_to_yolo",
    "bbox_xywh_to_cxcywhn",
    "bbox_xywh_to_xyxy",
    "bbox_xywh_to_xyxyn",
    "bbox_xyxy_to_cxcywhn",
    "bbox_xyxy_to_xywh",
    "bbox_xyxy_to_xywh",
    "bbox_xyxy_to_xyxyn",
    "bbox_xyxyn_to_cxcywhn",
    "bbox_xyxyn_to_xywh",
    "bbox_xyxyn_to_xyxy",
    "bbox_yolo_to_coco",
    "bbox_yolo_to_voc",
    "convert_bbox",
    "get_enclosing_bbox",
]

import numpy as np

from mon.globals import ShapeCode


# region Property

def bbox_area(bbox: np.ndarray) -> np.ndarray:
    """Compute the area(s) of bounding box(es).
    
    Args:
        bbox: Bounding box(es) of shape `[4]` or `[N, 4]` and in
            XYXY format.
    
    Returns:
        A `[11]`, or an `[N]` array containing the area value(s).
    """
    if bbox.ndim == 1:
        bbox = np.expand_dims(bbox, 0)
    assert bbox.ndim == 2, f"`bbox` must be 1D, but got {bbox.ndim}D."
    x1   = bbox[..., 0]
    y1   = bbox[..., 1]
    x2   = bbox[..., 2]
    y2   = bbox[..., 3]
    area = (x2 - x1) * (y2 - y1)
    return area


def bbox_center(bbox: np.ndarray) -> np.ndarray:
    """Compute the center(s) of bounding box(es).
    
    Args:
        bbox: Bounding box(es) of shape `[4]` or `[N, 4]` and in
            XYXY format.
    
    Returns:
        An `[1, 2]`, or an `[N, 2]` array containing the center(s) of
        bounding box(es) in `[cx, cy]` format.
    """
    if bbox.ndim == 1:
        bbox = np.expand_dims(bbox, 0)
    assert bbox.ndim == 2, f"`bbox` must be 1D, but got {bbox.ndim}D."
    x1     = bbox[..., 0]
    y1     = bbox[..., 1]
    x2     = bbox[..., 2]
    y2     = bbox[..., 3]
    cx     = x1 + (x2 - x1) / 2.0
    cy     = y1 + (y2 - y1) / 2.0
    center = np.stack((cx, cy), -1)
    return center


def bbox_corners(bbox: np.ndarray) -> np.ndarray:
    """Get corner(s) of bounding box(es) in an array.
    
    Args:
        bbox: Bounding box(es) of shape `[4]` or `[N, 4]` and in
            XYXY format.
    
    Returns:
        An `[1, 8]`, or an `[N, 8]` array containing corners of
        bounding box(es) in `[x1, y1, x2, y2, x3, y3, x4, y4]` format.
    """
    if bbox.ndim == 1:
        bbox = np.expand_dims(bbox, 0)
    assert bbox.ndim == 2, f"`bbox` must be 1D, but got {bbox.ndim}D."
    x1      = bbox[..., 0]
    y1      = bbox[..., 1]
    x2      = bbox[..., 2]
    y2      = bbox[..., 3]
    w       = x2 - x1
    h       = y2 - y1
    c_x1    = x1
    c_y1    = y1
    c_x2    = x1 + w
    c_y2    = y1
    c_x3    = x2
    c_y3    = y2
    c_x4    = x1
    c_y4    = y1 + h
    corners = np.hstack((c_x1, c_y1, c_x2, c_y2, c_x3, c_y3, c_x4, c_y4))
    return corners


def bbox_corners_points(bbox: np.ndarray) -> np.ndarray:
    """Get corner(s) of bounding box(es) as points.
    
    Args:
        bbox: Bounding box(es) of shape `[4]` or `[N, 4]` and in
            XYXY format.
    
    Returns:
        An `[1, 4, 2]`, or an `[N, 4, 2]` array containing corners of
        bounding box(es) in `[[x1, y1], [x2, y2], [x3, y3], [x4, y4]]` format.
    """
    if bbox.ndim == 1:
        bbox = np.expand_dims(bbox, 0)
    assert bbox.ndim == 2, f"`bbox` must be 1D, but got {bbox.ndim}D."
    x1     = bbox[..., 0]
    y1     = bbox[..., 1]
    x2     = bbox[..., 2]
    y2     = bbox[..., 3]
    w      = x2 - x1
    h      = y2 - y1
    c_x1   = x1
    c_y1   = y1
    c_x2   = x1 + w
    c_y2   = y1
    c_x3   = x2
    c_y3   = y2
    c_x4   = x1
    c_y4   = y1 + h
    points = np.array([[c_x1, c_y1], [c_x2, c_y2], [c_x3, c_y3], [c_x4, c_y4]], np.int32)
    return points


def get_enclosing_bbox(bbox: np.ndarray) -> np.ndarray:
    """Get the enclosing bounding box(es) for rotated corners.
    
    Args:
        bbox: Bounding of shape [..., 8], containing N bounding boxes each
            described by their corner co-ordinates (x1 y1 x2 y2 x3 y3 x4 y4).

    Returns:
        Bounding boxes of shape [..., 4] and in XYXY format.
    """
    x_ = bbox[:, [0, 2, 4, 6]]
    y_ = bbox[:, [1, 3, 5, 7]]
    x1 = np.min(x_, 1).reshape(-1, 1)
    y1 = np.min(y_, 1).reshape(-1, 1)
    x2 = np.max(x_, 1).reshape(-1, 1)
    y2 = np.max(y_, 1).reshape(-1, 1)
    return np.hstack((x1, y1, x2, y2, bbox[:, 8:]))

# endregion


# region Association

def bbox_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> np.ndarray:
    """Compute the intersection-over-union (Jaccard index) between two (sets) of
    bounding box(es).
    
    Args:
        bbox1: Predicted bounding box(es) of shape `[4]` or `[N, 4]`
            and in XYXY format.
        bbox2: Ground-truth bounding box(es) of shape `[4]` or `[M, 4]`
            and in XYXY format.
    
    Returns:
        An `NxM` matrix containing the pairwise IoU values.
    """
    # Make sure the bboxes are in 2D arrays.
    if bbox1.ndim == 1:
        bbox1 = np.expand_dims(bbox1, 0)
    if bbox2.ndim == 1:
        bbox2 = np.expand_dims(bbox2, 0)
    assert bbox1.ndim == 2, f"`bbox1` must be 1D, but got {bbox1.ndim}D."
    assert bbox2.ndim == 2, f"`bbox2` must be 1D, but got {bbox2.ndim}D."
    # Expand the dimensions of the bboxes to calculate pairwise IoU values.
    bbox1 = np.expand_dims(bbox1, 1)
    bbox2 = np.expand_dims(bbox2, 0)
    # IoU calculation.
    xx1   = np.maximum(bbox1[..., 0], bbox2[..., 0])
    yy1   = np.maximum(bbox1[..., 1], bbox2[..., 1])
    xx2   = np.minimum(bbox1[..., 2], bbox2[..., 2])
    yy2   = np.minimum(bbox1[..., 3], bbox2[..., 3])
    w     = np.maximum(0.0, xx2 - xx1)
    h     = np.maximum(0.0, yy2 - yy1)
    wh    = w * h
    union = ((bbox1[..., 2] - bbox1[..., 0]) * (bbox1[..., 3] - bbox1[..., 1])
           + (bbox2[..., 2] - bbox2[..., 0]) * (bbox2[..., 3] - bbox2[..., 1]) - wh)
    iou   = wh / union
    return iou


def bbox_giou(bbox1: np.ndarray, bbox2: np.ndarray) -> np.ndarray:
    """Compute the generalized intersection-over-union (Jaccard index) between
    two (sets) of bounding box(es).
    
    Args:
        bbox1: Predicted bounding box(es) of shape `[4]` or `[N, 4]`
            and in XYXY format.
        bbox2: Ground-truth bounding box(es) of shape `[4]` or `[M, 4]`
            and in XYXY format.
    
    Returns:
        An `NxM` matrix containing the pairwise IoU values.
    
    References:
        `<https://arxiv.org/pdf/1902.09630.pdf>`__
    """
    # Make sure the bboxes are in 2D arrays.
    if bbox1.ndim == 1:
        bbox1 = np.expand_dims(bbox1, 0)
    if bbox2.ndim == 1:
        bbox2 = np.expand_dims(bbox2, 0)
    assert bbox1.ndim == 2, f"`bbox1` must be 1D, but got {bbox1.ndim}D."
    assert bbox2.ndim == 2, f"`bbox2` must be 1D, but got {bbox2.ndim}D."
    # Expand the dimensions of the bboxes to calculate pairwise IoU values.
    bbox1 = np.expand_dims(bbox1, 1)
    bbox2 = np.expand_dims(bbox2, 0)
    # IoU calculation.
    xx1   = np.maximum(bbox1[..., 0], bbox2[..., 0])
    yy1   = np.maximum(bbox1[..., 1], bbox2[..., 1])
    xx2   = np.minimum(bbox1[..., 2], bbox2[..., 2])
    yy2   = np.minimum(bbox1[..., 3], bbox2[..., 3])
    w     = np.maximum(0.0, xx2 - xx1)
    h     = np.maximum(0.0, yy2 - yy1)
    wh    = w * h
    union = ((bbox1[..., 2] - bbox1[..., 0]) * (bbox1[..., 3] - bbox1[..., 1])
             + (bbox2[..., 2] - bbox2[..., 0]) * (bbox2[..., 3] - bbox2[..., 1]) - wh)
    iou   = wh / union
    
    xxc1  = np.minimum(bbox1[..., 0], bbox2[..., 0])
    yyc1  = np.minimum(bbox1[..., 1], bbox2[..., 1])
    xxc2  = np.maximum(bbox1[..., 2], bbox2[..., 2])
    yyc2  = np.maximum(bbox1[..., 3], bbox2[..., 3])
    wc    = xxc2 - xxc1
    hc    = yyc2 - yyc1
    assert (wc > 0).all() and (hc > 0).all()
    area_enclose = wc * hc
    giou  = iou - (area_enclose - union) / area_enclose
    giou  = (giou + 1.0) / 2.0  # resize from (-1,1) to (0,1)
    return giou


def bbox_diou(bbox1: np.ndarray, bbox2: np.ndarray) -> np.ndarray:
    """Compute the distance intersection-over-union (Jaccard index) between
    two (sets) of bounding box(es).
    
    Args:
        bbox1: Predicted bounding box(es) of shape `[4]` or `[N, 4]`
            and in XYXY format.
        bbox2: Ground-truth bounding box(es) of shape `[4]` or `[M, 4]`
            and in XYXY format.
    
    Returns:
        An `NxM` matrix containing the pairwise IoU values.
    
    References:
        `<https://arxiv.org/pdf/1902.09630.pdf>`__
    """
    # Make sure the bboxes are in 2D arrays.
    if bbox1.ndim == 1:
        bbox1 = np.expand_dims(bbox1, 0)
    if bbox2.ndim == 1:
        bbox2 = np.expand_dims(bbox2, 0)
    assert bbox1.ndim == 2, f"`bbox1` must be 1D, but got {bbox1.ndim}D."
    assert bbox2.ndim == 2, f"`bbox2` must be 1D, but got {bbox2.ndim}D."
    # Expand the dimensions of the bboxes to calculate pairwise IoU values.
    bbox1 = np.expand_dims(bbox1, 1)
    bbox2 = np.expand_dims(bbox2, 0)
    # IoU calculation.
    xx1   = np.maximum(bbox1[..., 0], bbox2[..., 0])
    yy1   = np.maximum(bbox1[..., 1], bbox2[..., 1])
    xx2   = np.minimum(bbox1[..., 2], bbox2[..., 2])
    yy2   = np.minimum(bbox1[..., 3], bbox2[..., 3])
    w     = np.maximum(0.0, xx2 - xx1)
    h     = np.maximum(0.0, yy2 - yy1)
    wh    = w * h
    union = ((bbox1[..., 2] - bbox1[..., 0]) * (bbox1[..., 3] - bbox1[..., 1])
             + (bbox2[..., 2] - bbox2[..., 0]) * (bbox2[..., 3] - bbox2[..., 1]) - wh)
    iou   = wh / union
    
    centerx1   = (bbox1[..., 0] + bbox1[..., 2]) / 2.0
    centery1   = (bbox1[..., 1] + bbox1[..., 3]) / 2.0
    centerx2   = (bbox2[..., 0] + bbox2[..., 2]) / 2.0
    centery2   = (bbox2[..., 1] + bbox2[..., 3]) / 2.0
    inner_diag = (centerx1 - centerx2) ** 2 + (centery1 - centery2) ** 2
    
    xxc1       = np.minimum(bbox1[..., 0], bbox2[..., 0])
    yyc1       = np.minimum(bbox1[..., 1], bbox2[..., 1])
    xxc2       = np.maximum(bbox1[..., 2], bbox2[..., 2])
    yyc2       = np.maximum(bbox1[..., 3], bbox2[..., 3])
    outer_diag = (xxc2 - xxc1) ** 2 + (yyc2 - yyc1) ** 2
    
    diou       = iou - inner_diag / outer_diag
    return (diou + 1) / 2.0  # resize from (-1,1) to (0,1)


def bbox_ciou(bbox1: np.ndarray, bbox2: np.ndarray) -> np.ndarray:
    """Compute the complete intersection-over-union (Jaccard index) between
    two (sets) of bounding box(es).
    
    Args:
        bbox1: Predicted bounding box(es) of shape `[4]` or `[N, 4]`
            and in XYXY format.
        bbox2: Ground-truth bounding box(es) of shape `[4]` or `[M, 4]`
            and in XYXY format.
    
    Returns:
        An `NxM` matrix containing the pairwise IoU values.
    
    References:
        `<https://arxiv.org/pdf/1902.09630.pdf>`__
    """
    # Make sure the bboxes are in 2D arrays.
    if bbox1.ndim == 1:
        bbox1 = np.expand_dims(bbox1, 0)
    if bbox2.ndim == 1:
        bbox2 = np.expand_dims(bbox2, 0)
    assert bbox1.ndim == 2, f"`bbox1` must be 1D, but got {bbox1.ndim}D."
    assert bbox2.ndim == 2, f"`bbox2` must be 1D, but got {bbox2.ndim}D."
    # Expand the dimensions of the bboxes to calculate pairwise IoU values.
    bbox1 = np.expand_dims(bbox1, 1)
    bbox2 = np.expand_dims(bbox2, 0)
    # IoU calculation.
    xx1   = np.maximum(bbox1[..., 0], bbox2[..., 0])
    yy1   = np.maximum(bbox1[..., 1], bbox2[..., 1])
    xx2   = np.minimum(bbox1[..., 2], bbox2[..., 2])
    yy2   = np.minimum(bbox1[..., 3], bbox2[..., 3])
    w     = np.maximum(0.0, xx2 - xx1)
    h     = np.maximum(0.0, yy2 - yy1)
    wh    = w * h
    union = ((bbox1[..., 2] - bbox1[..., 0]) * (bbox1[..., 3] - bbox1[..., 1])
             + (bbox2[..., 2] - bbox2[..., 0]) * (bbox2[..., 3] - bbox2[..., 1]) - wh)
    iou   = wh / union
    
    centerx1   = (bbox1[..., 0] + bbox1[..., 2]) / 2.0
    centery1   = (bbox1[..., 1] + bbox1[..., 3]) / 2.0
    centerx2   = (bbox2[..., 0] + bbox2[..., 2]) / 2.0
    centery2   = (bbox2[..., 1] + bbox2[..., 3]) / 2.0
    inner_diag = (centerx1 - centerx2) ** 2 + (centery1 - centery2) ** 2
    
    xxc1       = np.minimum(bbox1[..., 0], bbox2[..., 0])
    yyc1       = np.minimum(bbox1[..., 1], bbox2[..., 1])
    xxc2       = np.maximum(bbox1[..., 2], bbox2[..., 2])
    yyc2       = np.maximum(bbox1[..., 3], bbox2[..., 3])
    outer_diag = (xxc2 - xxc1) ** 2 + (yyc2 - yyc1) ** 2
    
    w1 = bbox1[..., 2] - bbox1[..., 0]
    h1 = bbox1[..., 3] - bbox1[..., 1]
    w2 = bbox2[..., 2] - bbox2[..., 0]
    h2 = bbox2[..., 3] - bbox2[..., 1]
   
    # Prevent dividing over zero. add one pixel shift
    h2     = h2 + 1.0
    h1     = h1 + 1.0
    arctan = np.arctan(w2 / h2) - np.arctan(w1 / h1)
    v      = (4 / (np.pi ** 2)) * (arctan ** 2)
    S      = 1 - iou
    alpha  = v / (S + v)
    ciou   = iou - inner_diag / outer_diag - alpha * v
    return (ciou + 1) / 2.0  # resize from (-1,1) to (0,1)


def bbox_center_distance(bbox1: np.ndarray, bbox2: np.ndarray) -> np.ndarray:
    """Measure the center distance(s) between two (sets) of bounding box(es).
    This is a coarse implementation, we don't recommend using it only for
    association, which can be unstable and sensitive to frame rate and object speed.
    
    Args:
        bbox1: Predicted bounding box(es) of shape `[4]` or `[N, 4]`
            and in XYXY format.
        bbox2: Ground-truth bounding box(es) of shape `[4]` or `[M, 4]`
            and in XYXY format.
    
    Returns:
        An `NxM` matrix containing the pairwise center distance value(s).
    """
    # Make sure the bboxes are in 2D arrays.
    if bbox1.ndim == 1:
        bbox1 = np.expand_dims(bbox1, 0)
    if bbox2.ndim == 1:
        bbox2 = np.expand_dims(bbox2, 0)
    assert bbox1.ndim == 2, f"`bbox1` must be 1D, but got {bbox1.ndim}D."
    assert bbox2.ndim == 2, f"`bbox2` must be 1D, but got {bbox2.ndim}D."
    # Expand the dimensions of the bboxes to calculate pairwise IoU values.
    bbox1    = np.expand_dims(bbox1, 1)
    bbox2    = np.expand_dims(bbox2, 0)
    centerx1 = (bbox1[..., 0] + bbox2[..., 2]) / 2.0
    centery1 = (bbox1[..., 1] + bbox2[..., 3]) / 2.0
    centerx2 = (bbox1[..., 0] + bbox2[..., 2]) / 2.0
    centery2 = (bbox1[..., 1] + bbox2[..., 3]) / 2.0
    ct_dist2 = (centerx1 - centerx2) ** 2 + (centery1 - centery2) ** 2
    ct_dist  = np.sqrt(ct_dist2)
    # The linear rescaling is a naive version and needs more study
    ct_dist  = ct_dist / ct_dist.max()
    return ct_dist.max() - ct_dist  # resize to (0,1)

# endregion


# region Conversion

def bbox_cxcywhn_to_xywh(bbox: np.ndarray, height: int, width: int) -> np.ndarray:
    """Convert bounding boxes from CXCYWHN format to XYWH format."""
    cx_norm, cy_norm, w_norm, h_norm, *_ = bbox.T
    w    = w_norm * width
    h    = h_norm * height
    x    = (cx_norm * width)  - (w / 2.0)
    y    = (cy_norm * height) - (h / 2.0)
    bbox = np.stack((x, y, w, h), axis=-1)
    return bbox


def bbox_cxcywhn_to_xyxy(bbox: np.ndarray, height: int, width: int) -> np.ndarray:
    """Convert bounding boxes from CXCYWHN format to XYXY format."""
    cx_norm, cy_norm, w_norm, h_norm, *_ = bbox.T
    x1   = width  * (cx_norm - w_norm / 2)
    y1   = height * (cy_norm - h_norm / 2)
    x2   = width  * (cx_norm + w_norm / 2)
    y2   = height * (cy_norm + h_norm / 2)
    bbox = np.stack((x1, y1, x2, y2), axis=-1)
    return bbox


def bbox_cxcywhn_to_xyxyn(bbox: np.ndarray) -> np.ndarray:
    """Convert bounding boxes from CXCYWHN format to XYXYN format."""
    cx_norm, cy_norm, w_norm, h_norm, *_ = bbox.T
    x1   = (cx_norm - w_norm / 2)
    y1   = (cy_norm - h_norm / 2)
    x2   = (cx_norm + w_norm / 2)
    y2   = (cy_norm + h_norm / 2)
    bbox = np.stack((x1, y1, x2, y2), axis=-1)
    return bbox


def bbox_xywh_to_cxcywhn(bbox: np.ndarray, height: int, width: int) -> np.ndarray:
    """Convert bounding boxes from XYWH format to CXCYWHN format."""
    x, y, w, h, *_ = bbox.T
    cx      = x + (w / 2.0)
    cy      = y + (h / 2.0)
    cx_norm = cx / width
    cy_norm = cy / height
    w_norm  = w / width
    h_norm  = h / height
    bbox    = np.stack((cx_norm, cy_norm, w_norm, h_norm), axis=-1)
    return bbox


def bbox_xywh_to_xyxy(bbox: np.ndarray) -> np.ndarray:
    """Convert bounding boxes from XYWH format to XYXY format."""
    x, y, w, h, *_ = bbox.T
    x2   = x + w
    y2   = y + h
    bbox = np.stack((x, y, x2, y2), axis=-1)
    return bbox


def bbox_xywh_to_xyxyn(bbox: np.ndarray, height: int, width: int) -> np.ndarray:
    """Convert bounding boxes from XYWH format to XYXYN format."""
    x, y, w, h, *_ = bbox.T
    x2      = x + w
    y2      = y + h
    x1_norm = x / width
    y1_norm = y / height
    x2_norm = x2 / width
    y2_norm = y2 / height
    bbox    = np.stack((x1_norm, y1_norm, x2_norm, y2_norm), axis=-1)
    return bbox


def bbox_xyxy_to_cxcywhn(bbox: np.ndarray, height: int, width: int) -> np.ndarray:
    """Convert bounding boxes from XYXY format to CXCYWHN format."""
    x1, y1, x2, y2, *_ = bbox.T
    w       = x2 - x1
    h       = y2 - y1
    cx      = x1 + (w / 2.0)
    cy      = y1 + (h / 2.0)
    cx_norm = cx / width
    cy_norm = cy / height
    w_norm  = w / width
    h_norm  = h / height
    bbox    = np.stack((cx_norm, cy_norm, w_norm, h_norm), axis=-1)
    return bbox


def bbox_xyxy_to_xywh(bbox: np.ndarray) -> np.ndarray:
    """Convert bounding boxes from XYXY format to XYWH format."""
    x1, y1, x2, y2, *_ = bbox.T
    w    = x2 - x1
    h    = y2 - y1
    bbox = np.stack((x1, y1, w, h), axis=-1)
    return bbox


def bbox_xyxy_to_xyxyn(bbox: np.ndarray, height: int, width: int) -> np.ndarray:
    """Convert bounding boxes from XYXY format to XYXYN format."""
    x1, y1, x2, y2, *_ = bbox.T
    x1_norm = x1 / width
    y1_norm = y1 / height
    x2_norm = x2 / width
    y2_norm = y2 / height
    bbox    = np.stack((x1_norm, y1_norm, x2_norm, y2_norm), axis=-1)
    return bbox


def bbox_xyxyn_to_cxcywhn(bbox: np.ndarray) -> np.ndarray:
    """Convert bounding boxes from XYXYN format to CXCYWHN format."""
    x1, y1, x2, y2, *_ = bbox.T
    w_norm  = x2 - x1
    h_norm  = y2 - y1
    cx_norm = x1 + (w_norm / 2.0)
    cy_norm = y1 + (h_norm / 2.0)
    bbox    = np.stack((cx_norm, cy_norm, w_norm, h_norm), axis=-1)
    return bbox


def bbox_xyxyn_to_xywh(bbox: np.ndarray, height: int, width: int) -> np.ndarray:
    """Convert bounding boxes from XYXYN format to XYWH format."""
    x1, y1, x2, y2, *_ = bbox.T
    x1   = x1 * width
    x2   = x2 * width
    y1   = y1 * height
    y2   = y2 * height
    w    = x2 - x1
    h    = y2 - y1
    bbox = np.stack((x1, y1, w, h), axis=-1)
    return bbox


def bbox_xyxyn_to_xyxy(bbox: np.ndarray, height: int, width: int) -> np.ndarray:
    """Convert bounding boxes from XYXYN format to XYXY format."""
    x1, y1, x2, y2, *_ = bbox.T
    x1   = x1 * width
    x2   = x2 * width
    y1   = y1 * height
    y2   = y2 * height
    bbox = np.stack((x1, y1, x2, y2), axis=-1)
    return bbox


bbox_coco_to_voc  = bbox_xywh_to_xyxy
bbox_coco_to_yolo = bbox_xywh_to_cxcywhn
bbox_voc_to_coco  = bbox_xyxy_to_xywh
bbox_voc_to_yolo  = bbox_xyxy_to_cxcywhn
bbox_yolo_to_coco = bbox_cxcywhn_to_xywh
bbox_yolo_to_voc  = bbox_cxcywhn_to_xyxy


def convert_bbox(
    bbox  : np.ndarray,
    code  : ShapeCode | int,
    height: int = None,
    width : int = None
) -> np.ndarray:
    """Convert bounding box."""
    code = ShapeCode.from_value(value=code)
    match code:
        case ShapeCode.SAME:
            return bbox
        case ShapeCode.VOC2COCO | ShapeCode.XYXY2XYWH:
            return bbox_voc_to_coco(bbox=bbox)
        case ShapeCode.VOC2YOLO | ShapeCode.XYXY2CXCYN:
            return bbox_voc_to_yolo(bbox=bbox, height=height, width=width)
        case ShapeCode.COCO2VOC | ShapeCode.XYWH2XYXY:
            return bbox_coco_to_voc(bbox=bbox)
        case ShapeCode.COCO2YOLO | ShapeCode.XYWH2CXCYN:
            return bbox_coco_to_yolo(bbox=bbox, height=height, width=width)
        case ShapeCode.YOLO2VOC | ShapeCode.CXCYN2XYXY:
            return bbox_yolo_to_voc(bbox=bbox, height=height, width=width)
        case ShapeCode.YOLO2COCO | ShapeCode.CXCYN2XYXY:
            return bbox_yolo_to_coco(bbox=bbox, height=height, width=width)
        case _:
            raise ValueError(f"{code}.")

# endregion
