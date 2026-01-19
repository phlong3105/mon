#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Image segmentation algorithms.

This package contains various image segmentation methods commonly used in
computer vision.

Notes:
    - Design Pattern: Component-Based Framework.
    - Goal: Define a collection of components that can be assembled to form
      concrete <Name> implementations.
    - Structure:
        ::

            segment/
            ├── __init__.py             # Exposes all
            ├── base.py                 # Base classes and mixins
            ├── comp/                   # Components
            │   ├── __init__.py
            │   └── ...
            ├── impl/                   # Implementations
            │   ├── __init__.py
            │   └── ...
            ├── adapters/               # Adapters
            │   ├── __init__.py
            │   └── ...
            └── utils.py                # Utilities and helpers
"""

from __future__ import annotations

__all__ = [
    "sam",
]

from .adapters import *
from .base import *
from .comp import *
from .impl import *
from .utils import *


# ==============================================================================
# region REGISTRY & FACTORY
# ==============================================================================


# endregion
