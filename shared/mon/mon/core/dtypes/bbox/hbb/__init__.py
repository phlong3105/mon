#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This package implements data structure and processing functions for horizontal
bounding boxes (HBBs).
"""

__all__ = [
    "HBBs",
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

from .core import HBBs
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
from .utils import (
    is_cxcywhn,
    is_normalized,
    is_xywh,
    is_xyxy,
)
