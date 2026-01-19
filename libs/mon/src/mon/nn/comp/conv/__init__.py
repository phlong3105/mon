#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Convolutional layers.

This package contains various convolutional layers - the basic building blocks
of convolutional neural networks.

Notes:
    - Design Pattern: Template Method.
    - Goal: Define a family of algorithms that share a common processing pipeline.
    - Structure:
        ::

            conv/
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
from .dsconv import *
from .ghost import *
from .mobileone import *
from .utils import *
from .zacn import *


# ==============================================================================
# region REGISTRY & FACTORY
# ==============================================================================


# endregion
