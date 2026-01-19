#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Normalization layers.

This package contains various normalization layers commonly used in convolutional
neural networks (CNNs) and deep learning models.

Notes:
    - Design Pattern: Template Method.
    - Goal: Define a family of algorithms that share a common processing pipeline.
    - Structure:
        ::

            norm/
            ├── __init__.py             # Registry and factory
            ├── api.py                  # External APIs
            ├── base.py                 # Base classes and mixins
            ├── ...
            └── utils.py                # Utilities and helpers
"""

from __future__ import annotations

from .api import *
from .base import *
from .batchnorm import *
from .instancenorm import *
from .pono_ms import *
from .utils import *

# ==============================================================================
# region REGISTRY & FACTORY
# ==============================================================================


# endregion
