#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Metrics.

This package contains various metrics for evaluating model performance.

Notes:
    - Design Pattern: Template Method.
    - Goal: Define a family of algorithms that share a common processing pipeline.
    - Structure:
        ::
        
            metric/
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
from .complexity import *
from .external import *
from .image import *
from .utils import *


# ==============================================================================
# region REGISTRY & FACTORY
# ==============================================================================


# endregion
