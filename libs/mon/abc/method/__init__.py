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
from .external import *
from .utils import *

# __all__ = []  


# ==============================================================================
# region REGISTRY & FACTORY
# ==============================================================================


# endregion
