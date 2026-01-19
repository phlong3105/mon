#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Complex data types.

This package contains custom and complex data types used across the project.

Notes:
    - Design Pattern: Multiple Toolkits.
    - Goal: Encapsulate multiple "Toolkits" for multiple data types.
    - Structure:
        ::

            dtypes/
            ├── __init__.py             # Unified entry point
            ├── base.py                 # Global base classes and mixins
            ├── toolkit/
            │   ├── __init__.py         # Exposes all
            │   ├── api.py              # External APIs
            │   ├── core.py             # Base classes and mixins
            │   ├── io.py               # I/O operations
            │   ├── ops.py              # Atomic operations
            │   ├── exec.py             # Execution logic
            │   └── debug.py            # Debugging utilities
            └── ...
"""

from __future__ import annotations

__all__ = [
    # Flat exposed APIs
    "BBox",
    "BBoxList",
    "Class",
    "ClassList",
    "Data",
    "DepthMap",
    "DeviceManagementMixin",
    "Frame",
    "Image",
    "InfraredMap",
    "Instance",
    "PersistentData",
    "Probabilities",
    "SemanticMask",
    "TensorOrArray",
    "VideoWriter",
    "VideoWriterCV",
    "Weights",
    "WeightsEnum",
    "WeightsType",
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
from .base import Data, DeviceManagementMixin, PersistentData
from .bbox import BBox, BBoxList
from .classes import Class, ClassList, Probabilities
from .depth import DepthMap
from .image import Image
from .instance import Instance
from .mask import SemanticMask
from .thermal import InfraredMap
from .video import Frame, VideoWriter, VideoWriterCV
from .weights import Weights, WeightsEnum, WeightsType
