#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A package for bounding boxes data type.

This package implements data structure and processing functions for bounding
boxes. It supports both horizontal bounding boxes (Bboxes) and oriented bounding
boxes (OBBs).

The default format of a bounding box is: <cx, cy, w, h, a, cls, ...>, where ...
can be any additional information such as confidence score or tracking ID.
For HBBs, the angle ``a`` is always ``0``.
"""

__all__ = [
    "BBox",
    "BBoxes",
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
    "load",
    "normalize",
    "pad_square",
    "split",
    "to_2d",
    "xywh_to_cxcywhn",
    "xywh_to_xyxy",
    "xyxy_to_cxcywhn",
    "xyxy_to_xywh",
]

from .core import BBox, BBoxes
from .io import load
from .processing import (
    area,
    center,
    center_distance,
    ciou,
    convert,
    corners,
    corners_pts,
    crop_center,
    crop_fit_square,
    cxcywhn_to_xywh,
    cxcywhn_to_xyxy,
    denormalize,
    diou,
    enclosing,
    filter_iou,
    giou,
    iou,
    iou_matrix,
    normalize,
    pad_square,
    split,
    to_2d,
    xywh_to_cxcywhn,
    xywh_to_xyxy,
    xyxy_to_cxcywhn,
    xyxy_to_xywh,
)
from .utils import is_cxcywhn, is_normalized, is_xywh, is_xyxy
