#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Operations.

This package contains atomic operations for various data types.

File Structure:
::

    ops/
    ├── __init__.py
    ├── audio/          # Audio processing operations
    ├── geometry/       # Geometric operations (e.g., transformations, projections)
    ├── image/          # Image processing operations
    ├── text/           # Text processing operations
    ├── video/          # Video processing operations
    └── draw.py/        # General drawing utilities
"""

from __future__ import annotations

from .audio import *
from .draw import *
from .geometry import *
from .image import *
from .text import *
from .video import *
