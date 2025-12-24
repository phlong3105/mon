#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""<Name> data type.

This package contains a "full-stack" toolkit for <Name> data, including data
structure, ingestion, analysis, atomic transformations, complex workflows, and
rendering utilities.

This is the "Toolkit" pattern. It is useful for encapsulating all related
functionalities for a specific data type or domain within a single package.

The "Toolkit" Pattern typically has the following structure:
    toolkit/
    ├── __init__.py    # Exposes all functionalities
    ├── core.py        # Base classes and mixins
    ├── io.py          # Ingestion & Retrieval – This module handles moving the raw bit
    ├── meta.py        # Analysis – Operations that return information about the data without changing it
    ├── ops.py         # Atomic Transformations – Pure functions that perform a single mathematical or structural change
    ├── proc.py        # Complex Workflows – Higher-level logic that might involve multiple atomic steps
    └── vis.py         # Rendering – For debugging and human interaction
"""

__all__ = []

from .core import *
from .io import *
from .meta import *
from .ops import *
from .proc import *
from .vis import *
