#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Bounding box data type.

This package contains a "full-stack" toolkit for bounding box data, including
data structure, ingestion, analysis, atomic transformations, complex workflows,
and rendering utilities.
"""

__all__ = [
    "BBox",
    "BBoxList",
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
    "draw",
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

from .core import *
from .io import *
from .meta import *
from .ops import *
from .proc import *
from .vis import *
