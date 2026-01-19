#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Linear layers.

This package contains various linear layers commonly used in MLP and deep
neural networks.

Notes:
    - Design Pattern: Template Method.
    - Goal: Define a family of algorithms that share a common processing pipeline.
    - Structure:
        ::

            linear/
            ├── __init__.py             # Registry and factory
            ├── api.py                  # External APIs
            ├── base.py                 # Base classes and mixins
            ├── ...
            └── utils.py                # Utilities and helpers
"""

from __future__ import annotations

from .api import *
from .base import *
from .depth_aware import *
from .repr import *
from .utils import *

# ==============================================================================
# region REGISTRY & FACTORY
# ==============================================================================


# endregion
