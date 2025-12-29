#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Mixins for additional data handling functionalities.

This package contains a "full-stack" mixins for training data containers.

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
            ├── ops.py         # Utility and algorithms
            ├── proc.py        # Workflow orchestration
            └── vis.py         # UI/UX rendering
"""

__all__ = [
    "BatchCollateMixin",
    "DataLoadMixin",
    "InputTargetLoadMixin",
    "MultimodalDataLoadMixin",
    "RegistrableMixin",
    "RootLoadMixin",
]

from .core import *
from .io import  (
    DataLoadMixin,
    InputTargetLoadMixin,
    MultimodalDataLoadMixin,
    RootLoadMixin,
)
from .meta import RegistrableMixin
from .ops import *
from .proc import BatchCollateMixin
from .vis import *
