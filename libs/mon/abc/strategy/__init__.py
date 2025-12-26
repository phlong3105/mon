#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""<Name> strategy.

This package contains a "full-stack" strategy pattern for <Name>, including
strategy definition, registration, resolution, and execution utilities.

Notes:
    - Design Pattern: Strategy Pattern.
    - Goal: Define a family of algorithms, encapsulate each one, and make them
      interchangeable at RUNTIME.
    - Structure:
        ::
        
            strategy/
            ├── __init__.py            # Exposes all
            ├── base.py                # Base classes and mixins
            ├── context.py             # Execution context
            ├── registry.py            # Registry and factory logic
            ├── utils.py               # Utility functions and helpers
            └── algorithms/            # Concrete implementations
                ├── __init__.py        # Exposes all algorithms
                ├── algorithm_a.py
                ├── algorithm_b.py
                ├── algorithm_c.py
                └── ...
"""

__all__ = []

from .algorithms import *
from .base import *
from .context import *
from .registry import *
from .utils import *
