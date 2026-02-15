#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Computer Vision.

This package contains computer vision functionalities of the ``mon`` package.

File Structure:
::

    mon.core/
    ├── __init__.py
    ├── feature/        # Traditional CV (Edges, Corners, SIFT)
    ├── geometry/       # Camera calibration & 3D projections
    ├── io/             # I/O operations
    ├── models/         # CV-specific architectures & backbones
    ├── ops/            # Atomic operations
    └── utils/          # General purpose helpers
"""

from __future__ import annotations

from .feature import *
from .geometry import *
from .io import *
from .models import *
from .ops import *
from .utils import *
