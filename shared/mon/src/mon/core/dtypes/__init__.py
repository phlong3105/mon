#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A package for core data types.

This package implements utility functions for type manipulation and conversion,
as well as defines core data types used throughout the library.
"""

__all__ = [
    # Flat exposed APIs
    "BBox",
    "BBoxes",
    "BaseTensorOrArray",
    "Data",
    "DepthMap",
    "Frame",
    "Image",
    "InfraredMap",
    "Instance",
    "Probs",
    "SemanticMask",
    "draw_bbox",
    "draw_trajectory",
    # Hierarchical exposed APIs
    "bbox",
    "contour",
    "depth",
    "image",
    "instance",
    "mask",
    "probs",
    "thermal",
    "video",
]

from .base import BaseTensorOrArray, Data
from .bbox import BBox, BBoxes
from .depth import DepthMap
from .image import Image
from .instance import Instance
from .mask import SemanticMask
from .probs import Probs
from .thermal import InfraredMap
from .video import Frame
from .visualize import draw_bbox, draw_trajectory
