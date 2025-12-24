#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""<Name> strategy.

This package contains a "full-stack" strategy pattern for <Name>, including
strategy definition, registration, resolution, and execution utilities.

This is the "Strategy" pattern. It is useful for scenarios where multiple
algorithms or behaviors can be swapped interchangeably at RUNTIME, allowing the
system to choose the most appropriate one based on context.

The "Strategy" Pattern typically has the following structure:
    strategy/
    ├── __init__.py                     # Exposes the Registry and Context
    ├── base.py                         # Abstract Base Classes (The Contract)
    ├── context.py                      # The "Executor" that runs the strategy
    ├── registry.py                     # Logic for @register and .get()
    ├── utils.py                        # Performance decorators & shared utils
    └── algorithms/                     # Concrete implementations
        ├── __init__.py                 # Auto-import algorithms here
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
