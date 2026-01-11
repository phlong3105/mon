#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Mixins for additional data handling functionalities.

This package contains a "full-stack" mixins for training data containers.

Notes:
    - Design Pattern: Toolkit Pattern.
    - Goal: Encapsulate related functionalities for a specific data type.
    - Structure:
        ::
        
            toolkit/
            ├── __init__.py    # Exposes all
            ├── api.py         # External APIs
            ├── core.py        # Base classes and mixins
            ├── io.py          # I/O operations
            ├── ops.py         # Atomic operations
            ├── proc.py        # Execution logic
            └── debug.py       # Debugging utilities
"""

from __future__ import annotations

from .core import *
from .debug import *
from .io import *
from .ops import *
from .proc import *
