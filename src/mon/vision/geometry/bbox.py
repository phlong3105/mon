#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements geometry functions for bounding boxes."""

from __future__ import annotations

__all__ = [
    "bbox_coco_to_voc", "bbox_coco_to_yolo", "bbox_cxcywhn_to_xywh",
    "bbox_cxcywhn_to_xyxy", "bbox_cxcywhn_to_xyxyn", "bbox_voc_to_coco",
    "bbox_voc_to_yolo", "bbox_xywh_to_cxcywhn", "bbox_xywh_to_xyxy",
    "bbox_xywh_to_xyxyn", "bbox_xyxy_to_cxcywhn", "bbox_xyxy_to_xywh",
    "bbox_xyxy_to_xywh", "bbox_xyxy_to_xyxyn", "bbox_xyxyn_to_cxcywhn",
    "bbox_xyxyn_to_xywh", "bbox_xyxyn_to_xyxy", "bbox_yolo_to_coco",
    "bbox_yolo_to_voc", "convert_bbox", "get_bbox_area", "get_bbox_center",
    "get_bbox_corners", "get_bbox_corners_points",
    "get_bbox_intersection_union", "get_bbox_iou", "get_enclosing_bbox",
    "get_single_bbox_iou",
]

import numpy as np
import torch

from mon.globals import ShapeCode
from mon.vision import image


# region Property

def get_bbox_area(bbox: np.ndarray) -> np.ndarray:
    """Compute the area of bounding boxes.
    
    Args:
        bbox: Bounding boxes in XYXY format.
    
    Returns:
        The area value for each bbox.
    """
    x1, y1, x2, y2, *_ = bbox.T
    area = (x2 - x1) * (y2 - y1)
    return area


def get_bbox_center(bbox: np.ndarray) -> np.ndarray:
    """Compute the center of bounding box(es).
    
    Args:
        bbox: Bounding boxes in XYXY format.
    
    Returns:
        The center for each bbox described by the coordinates (cx, cy).
    """
    x1, y1, x2, y2, *_ = bbox.T
    cx     = x1 + (x2 - x1) / 2
    cy     = y1 + (y2 - y1) / 2
    center = np.stack((cx, cy), -1)
    return center


def get_bbox_corners(bbox: np.ndarray) -> np.ndarray:
    """Get corners of bounding boxes.
    
    Args:
        bbox: Bounding boxes in XYXY format.
    
    Returns:
        A tensor of shape `N x 8` containing N bounding boxes each described by
        their corner coordinates (x1 y1 x2 y2 x3 y3 x4 y4).
    """
    x1, y1, x2, y2, *_ = bbox.T
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


def get_bbox_corners_points(bbox: np.ndarray) -> np.ndarray:
    """Get corners of bounding boxes as points.
    
    Args:
        bbox: Bounding boxes in XYXY format.
    
    Returns:
        Corners.
    """
    x1, y1, x2, y2, *_ = bbox.T
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


def get_bbox_intersection_union(
    bbox1: np.ndarray,
    bbox2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the intersection and union of two sets of boxes.
    
    References:
        https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
    
    Args:
        bbox1: The first set of boxes in XYXY format.
        bbox2: The second set of boxes in XYXY format.
        
    Returns:
        Intersection.
        Union.
    """
    area1 = get_bbox_area(bbox=bbox1)
    area2 = get_bbox_area(bbox=bbox2)
    lt    = np.max(bbox1[:, None, :2], bbox2[:, :2])  # [N, M, 2]
    rb    = np.min(bbox1[:, None, 2:], bbox2[:, 2:])  # [N, M, 2]
    wh    = image.upcast(rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]
    union = area1[:, None] + area2 - inter
    return inter, union


def get_bbox_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> np.ndarray:
    """Return the intersection-over-union (Jaccard index) between two sets of
    boxes.
    
    Args:
        bbox1: The first set of boxes in XYXY format.
        bbox2: The second set of boxes of in XYXY format.
    
    Returns:
        The NxM matrix containing the pairwise IoU values for every element
        in :param:`bbox1` and :param:`bbox2`.
    """
    inter, union = get_bbox_intersection_union(bbox1=bbox1, bbox2=bbox2)
    iou = inter / union
    return iou


def get_single_bbox_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> np.ndarray | float:
    """Return intersection-over-union (Jaccard index) between two boxes.
    
    Args:
        bbox1: The first bbox of shape in XYXY format.
        bbox2: The second bbox of shape in XYXY format.
    
    Returns:
        The IoU value for :param:`bbox1` and :param:`bbox2`.
    """
    xx1 = np.maximum(bbox1[0], bbox2[0])
    yy1 = np.maximum(bbox1[1], bbox2[1])
    xx2 = np.minimum(bbox1[2], bbox2[2])
    yy2 = np.minimum(bbox1[3], bbox2[3])
    w   = np.maximum(0.0, xx2 - xx1)
    h   = np.maximum(0.0, yy2 - yy1)
    wh  = w * h
    iou = wh / ((bbox1[2] - bbox1[0]) *
                (bbox1[3] - bbox1[1]) +
                (bbox2[2] - bbox2[0]) *
                (bbox2[3] - bbox2[1]) - wh)
    return iou


def get_enclosing_bbox(bbox: np.ndarray) -> np.ndarray:
    """Get an enclosing bbox for rotated corners of a bounding box.
    
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


# region Conversion

def bbox_cxcywhn_to_xywh(bbox: np.ndarray, height: int, width: int) -> np.ndarray:
    """Convert bounding boxes from CXCYWHN format to XYWH format."""
    cx_norm, cy_norm, w_norm, h_norm, *_ = bbox.T
    w    = w_norm * width
    h    = h_norm * height
    x    = (cx_norm * width) - (w / 2.0)
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
    height: int | None = None,
    width : int | None = None
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


# region Affine Transform

def clip_bbox(
    bbox      : np.ndarray,
    image_size: int | list[int],
    drop_ratio: float = 0.0,
) -> np.ndarray:
    """Clip bounding boxes to an image size and removes the bounding boxes,
    which lose too much area as a result of the augmentation.
    
    Args:
        bbox: Bounding boxes of shape [..., 4] and in XYXY format.
        image_size: An image size in HW format.
        drop_ratio: If the fraction of a bounding box left in the image after
            being clipped is less than :param:`drop_ratio` the bounding box is
            dropped. If :param:`drop_ratio` == 0, don't drop any bounding boxes.
            Defaults to 0.0.
        
    Returns:
        Clipped bounding boxes of shape [N, 4].
    """
    h, w       = image.get_hw(size=image_size)
    area       = get_bbox_area(bbox=bbox)
    bbox[:, 0] = np.clip(0, w)  # x1
    bbox[:, 1] = np.clip(0, h)  # y1
    bbox[:, 2] = np.clip(0, w)  # x2
    bbox[:, 3] = np.clip(0, h)  # y2
    new_area   = get_bbox_area(bbox=bbox)
    delta_area = (area - new_area) / area
    mask       = (delta_area < (1 - drop_ratio)).to(torch.int)
    bbox       = bbox[mask == 1, :]
    return bbox

# endregion
