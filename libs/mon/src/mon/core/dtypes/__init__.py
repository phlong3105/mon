#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A collection of custom data types.

This package provides custom and complex data types used across the project.
This package exposes common, flat aliases for frequent types and groups
domain-specific implementations under subpackages. This package enables
consistent data representation and manipulation for downstream modules.

Package structure:
    dtypes/
    ├── __init__.py           # Unified entry point
    ├── base.py               # Global BaseDType (abstract)
    ├── mixins/               # Shared Mixins (Registrable, IO, etc.)
    ├── image/                # Image sub-package
    │   ├── __init__.py
    │   ├── core.py           # Base classes and mixins
    │   ├── io.py             # Ingestion & Retrieval – This module handles moving the raw bit
    │   ├── meta.py           # Analysis – Operations that return information about the data without changing it
    │   ├── ops.py            # Atomic Transformations – Pure functions that perform a single mathematical or structural change
    │   ├── proc.py           # Complex Workflows – Higher-level logic that might involve multiple atomic steps
    │   └── vis.py            # Rendering – For debugging and human interaction
    └── ... (contour, depth, etc.)
"""

__all__ = [
    # Flat exposed APIs
    "BBox",
    "BBoxList",
    "TensorOrArray",
    "Class",
    "ClassList",
    "Data",
    "DepthMap",
    "Frame",
    "Image",
    "InfraredMap",
    "Instance",
    "SemanticMask",
    # Hierarchical exposed APIs
    "bbox",
    "contour",
    "depth",
    "image",
    "instance",
    "mask",
    "thermal",
    "video",
]

from .array import TensorOrArray
from .base import Data
from .bbox import BBox, BBoxList
from .classes import Class, ClassList
from .depth import DepthMap
from .image import Image
from .instance import Instance
from .mask import SemanticMask
from .thermal import InfraredMap
from .video import Frame
