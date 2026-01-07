#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A collection of complex data types.

This package provides custom and complex data types used across the project.
This package exposes common, flat aliases for frequent types and groups
domain-specific implementations under subpackages. This package enables
consistent data representation and manipulation for downstream modules.

Notes:
    - Design Pattern: Multiple Toolkits Pattern.
    - Goal: Encapsulate multiple "Toolkits" for multiple data types.
    - Structure:
        ::
        
            dtypes/
            ├── __init__.py        # Unified entry point
            ├── base.py            # Global base classes and mixins
            ├── toolkit/           # A "Toolkit" for a specific data type
            │   ├── __init__.py    # Exposes all
            │   ├── core.py        # Base classes and mixins
            │   ├── io.py          # Resource management
            │   ├── meta.py        # Discovery and lookup
            │   ├── ops.py         # Utility and algorithm
            │   ├── proc.py        # Workflow orchestration
            │   └── vis.py         # UI/UX rendering
            └── ... (contour, depth, etc.)
"""

__all__ = [
    # Flat exposed APIs
    "BBox",
    "BBoxList",
    "Class",
    "ClassList",
    "Data",
    "DataLoadMixin",
    "DepthMap",
    "DeviceManagementMixin",
    "Frame",
    "Image",
    "InfraredMap",
    "Instance",
    "Probabilities",
    "SemanticMask",
    "TensorOrArray",
    "VideoWriter",
    "VideoWriterCV",
    "VideoWriterFFmpeg",
    "Weights",
    "WeightsEnum",
    # Hierarchical exposed APIs
    "array",
    "bbox",
    "classes",
    "contour",
    "depth",
    "image",
    "instance",
    "mask",
    "thermal",
    "video",
    "weights",
]

from .array import TensorOrArray
from .base import Data, DataLoadMixin, DeviceManagementMixin
from .bbox import BBox, BBoxList
from .classes import Class, ClassList, Probabilities
from .depth import DepthMap
from .image import Image
from .instance import Instance
from .mask import SemanticMask
from .thermal import InfraredMap
from .video import Frame, VideoWriter, VideoWriterCV, VideoWriterFFmpeg
from .weights import Weights, WeightsEnum
