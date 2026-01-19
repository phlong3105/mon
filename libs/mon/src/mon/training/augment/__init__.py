#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Data augmentation.

This package contains various data augmentation and transformation techniques
used in training machine learning models.

Notes:
    - Design Pattern: Component-Based Framework.
    - Goal: Define a collection of components that can be assembled to form
      concrete data augmentation implementations.
    - Structure:
        ::

            augment/
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

from .adapters import *
from .base import *
from .comp import *
from .comp import albumentations
from .impl import *
from .utils import *


# ==============================================================================
# region REGISTRY & FACTORY
# ==============================================================================


# endregion
