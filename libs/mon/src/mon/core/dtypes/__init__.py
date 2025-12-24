#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A collection of custom data types.

This package provides custom and complex data types used across the project.
This package exposes common, flat aliases for frequent types and groups
domain-specific implementations under subpackages. This package enables
consistent data representation and manipulation for downstream modules.

Package structure:
    dtypes/
    ├── __init__.py        # Unified entry point
    ├── base.py            # Global BaseDType (abstract)
    ├── abc/               #
    │   ├── __init__.py    
    │   ├── core.py        # Base classes and mixins
    │   ├── io.py          # Ingestion & Retrieval – This module handles moving the raw bit
    │   ├── meta.py        # Analysis – Operations that return information about the data without changing it
    │   ├── ops.py         # Atomic Transformations – Pure functions that perform a single mathematical or structural change
    │   ├── proc.py        # Complex Workflows – Higher-level logic that might involve multiple atomic steps
    │   └── vis.py         # Rendering – For debugging and human interaction
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
