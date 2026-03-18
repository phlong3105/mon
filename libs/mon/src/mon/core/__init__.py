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
    ├── console.py      # Console utilities and logging
    ├── constants.py    # Global hardcoded values
    ├── context.py      # System context
    ├── dtype.py        # Global Enums
    ├── factory.py      # Registry and factory design patterns
    ├── fileio.py       # Atomic and optimized read/write operations
    ├── filesystem.py   # Filesystem operations
    ├── path.py         # Path manipulation
    ├── typing.py       # Type hints
    └── utils.py        # General purpose helpers
"""

from __future__ import annotations

from .base import *
from .config import *
from .console import *
from .constants import *
from .context import *
from .data import *
from .dtype import *
from .factory import *
from .fileio import *
from .filesystem import *
from .path import *
from .typing import *
from .ui import *
from .utils import *
