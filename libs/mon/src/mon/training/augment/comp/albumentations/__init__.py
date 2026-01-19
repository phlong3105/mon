#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Albumentations-based data augmentation and transformation.

This package contains various data augmentations and transformations using the
``albumentations`` library.

Notes:
    - Design Pattern: Template Method.
    - Goal: Define a family of algorithms that share a common processing pipeline.
    - Structure:
        ::

            albumentations/
            ├── __init__.py             # Registry and factory
            ├── api.py                  # External APIs
            ├── base.py                 # Base classes and mixins
            ├── basic.py                # Basic functionalities
            ├── ...
            ├── utils.py                # Utilities and helpers
            └── external/               # Integrate external libraries
                └── ...
"""

from __future__ import annotations

from .api import *
from .base import *
from .ftt import *
from .ifish import *
from .pixel import *
from .resize import *
from .utils import *


# ==============================================================================
# region REGISTRY & FACTORY
# ==============================================================================


# endregion
