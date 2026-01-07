#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Neural network's pre-trained weights.

This package contains a "full-stack" toolkit for pre-trained weights, including
data structure, ingestion, analysis, atomic transformations, complex workflows,
and rendering utilities.

Notes:
    - Design Pattern: Toolkit Pattern.
    - Goal: Encapsulate related functionalities for a specific data type or
      domain within a single package.
    - Structure:
        ::

            toolkit/           # A "Toolkit" for a specific data type
            ├── __init__.py    # Exposes all
            ├── core.py        # Base classes and mixins
            ├── io.py          # Resource management
            ├── meta.py        # Discovery and lookup
            ├── ops.py         # Utility and algorithm
            ├── proc.py        # Workflow orchestration
            └── vis.py         # UI/UX rendering
"""

# __all__ = []  # Prevent accidental imports of submodules.

from .core import *
from .io import *
from .meta import *
from .ops import *
from .proc import *
from .vis import *

