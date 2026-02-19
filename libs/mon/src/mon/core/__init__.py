#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Core.

This package contains the core functionalities of the ``mon`` package.

File Structure:
::

    mon.core/
    ├── __init__.py
    ├── base/           # Generic data structures
    ├── config/         # Configuration
    ├── data/           # Domain-specific data structures
    ├── ui/             # UI components
    ├── constants.py    # Global hardcoded values
    ├── dtype.py        # Global Enums
    ├── factory.py      # Registry and factory design patterns
    ├── fileio.py       # Atomic and optimized read/write operations
    ├── filesystem.py   # Filesystem operations
    ├── path.py         # Path manipulation
    ├── system.py       # Seeding, environment auditing, and process control
    ├── typing.py       # Type hints
    └── utils.py        # General purpose helpers
"""


from __future__ import annotations

from .base import *
from .config import *
from .constants import *
from .data import *
from .dtype import *
from .factory import *
from .fileio import *
from .filesystem import *
from .path import *
from .system import *
from .typing import *
from .ui import *
from .utils import *
