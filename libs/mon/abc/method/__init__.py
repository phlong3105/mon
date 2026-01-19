#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""<Name>.

This package contains various <name> commonly used in <domain>.

Notes:
    - Design Pattern: Template Method.
    - Goal: Define a family of algorithms that share a common processing pipeline.
    - Structure:
        ::

            template/
            ├── __init__.py             # Registry and factory
            ├── api.py                  # External APIs
            ├── base.py                 # Base classes and mixins
            ├── ...
            └── utils.py                # Utilities and helpers
"""

from __future__ import annotations

from .api import *
from .base import *
from .utils import *


# ==============================================================================
# region REGISTRY & FACTORY
# ==============================================================================


# endregion
