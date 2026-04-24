#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Image Operations.

This package contains operations for image processing.

File Structure:
::

    image/
    ├── __init__.py
    ├── color.py        # Color space conversions and manipulations
    ├── denoise.py      # Image denoising techniques
    ├── features.py     # Feature detection and extraction (e.g., edges, corners)
    ├── filter.py       # Image filtering operations (e.g., blurring, sharpening
    ├── io.py           # Image I/O operations (e.g., reading, writing, resizing)
    ├── noise.py        # Noise generation and analysis
    ├── patch.py        # Image patch extraction and manipulation
    ├── proc.py         # General image processing operations (e.g., geometric transformations, morphological operations)
    └── sample.py       # Image sampling and augmentation techniques
"""

from __future__ import annotations

from .color import *
from .denoise import *
from .features import *
from .filter import *
from .io import *
from .noise import *
from .patch import *
from .proc import *
from .sample import *
