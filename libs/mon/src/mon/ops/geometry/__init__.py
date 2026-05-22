#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Geometry Operations.

This package contains operations for geometric processing.

File Structure:
::

    geometry/
    ├── __init__.py
    ├── bbox.py         # Bounding box operations (e.g., IoU, NMS)
    ├── io.py           # I/O operations for geometric data (e.g., reading/writing annotations)
    ├── mask.py         # Mask operations (e.g., mask encoding/decoding, mask IoU)
    └── point.py/       # Point operations (e.g., distance calculations, point cloud processing)
"""

from __future__ import annotations

from .bbox import *
from .io import *
from .keypoint import *
from .mask import *
