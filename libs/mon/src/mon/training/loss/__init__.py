#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Loss functions.

This package contains various loss functions commonly used in machine learning
and deep learning. Each loss function is implemented as a class that inherits
from a common base class, allowing for easy integration and extension.

Notes:
    - Design Pattern: Template Method.
    - Goal: Define a family of algorithms that share a common processing pipeline.
    - Structure:
        ::

            loss/
            ├── __init__.py   # Registry and factory
            ├── api.py        # External APIs
            ├── base.py       # Base classes and mixins
            ├── basic.py      # Basic functionalities
            ├── ...
            ├── utils.py      # Utilities and helpers
            └── external/     # Integrate external libraries
                └── ...
"""

from __future__ import annotations

from .base import *
from .basic import *
from .extern import *
from .image import *
from .utils import *


# ==============================================================================
# region REGISTRY & FACTORY
# ==============================================================================


# endregion
