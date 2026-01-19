#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Neural network components.

This package contains various neural network components for building deep
learning models.

References:
    - Definition: https://www.ibm.com/think/topics/deep-learning#763338456

Notes:
    - Design Pattern: Component-Based Framework.
    - Goal: Define a collection of components that can be assembled to form
      concrete neural network implementations.
    - Structure:
        ::

            nn/
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

    - In this package, we follow the same coding conventions as PyTorch to
      maintain consistency. If you don't know what to do, just look at the
      PyTorch source code.
"""

from __future__ import annotations

__all__ = []

from .adapters import *
from .base import *
from .comp import *
from .comp import (
    act,
    attention,
    conv,
    dropout,
    fusion,
    linear,
    norm,
    padding,
    pooling,
)
from .impl import *
from .impl import inr
from .utils import *


# ==============================================================================
# region REGISTRY & FACTORY
# ==============================================================================


# endregion
