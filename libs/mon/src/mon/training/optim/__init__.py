#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Optimizers and learning rate schedulers.

This package contains various optimizers and learning rate schedulers used for
training machine learning models.

Notes:
    - Design Pattern: Template Method.
    - Goal: Define a family of algorithms that share a common processing pipeline.
    - Structure:
        ::

            optim/
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
