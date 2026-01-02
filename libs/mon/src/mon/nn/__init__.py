#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Neural network components.

This package contains various neural network components for building deep
learning models.

References:
    - Definition: https://www.ibm.com/think/topics/deep-learning#763338456
    
Notes:
    - Design Pattern: Component-Based Framework.
    - Goal: Build systems from reusable, interchangeable components that can be
      independently developed, tested, and maintained. Each component is a modular
      unit with well-defined interfaces, encapsulating specific functionality
      that can be assembled, replaced, or reused across applications.
    - Structure:
        ::
            
            nn/
            ├── __init__.py             # Exposes all
            ├── base.py                 # Base classes and mixins
            ├── comp/                   # Reusable components
            │   ├── __init__.py
            │   ├── base.py             # Component base classes and mixins
            │   └── ...                 # Concrete component
            ├── impl/                   # Concrete classes using base + components
            │   ├── __init__.py
            │   ├── concrete_impl.py    # Example implementation
            │   └── ...
            ├── usages/                 # Example usages of concrete implementations
            │   ├── __init__.py
            │   └── ...
            └── utils.py                # Utility functions and helpers
            
    - In this package, we follow the same coding conventions as PyTorch to
      maintain consistency. If you don't know what to do, just look at the
      PyTorch source code.
"""

# __all__ = []  # Prevent accidental imports of submodules.

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
from .usages import *
from .utils import *


# ==============================================================================
# REGISTRY & FACTORY (Type Resolution)
# ==============================================================================

# --- Register (Adding new spokes to the hub) ---


# --- Resolve (Retrieving spokes by name/key) ---
