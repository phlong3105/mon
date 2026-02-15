#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Core.

This package contains the core functionalities of the ``mon`` package.

File Structure:
::

    mon.core/
    ├── __init__.py
    ├── data/           # Generic containers
    ├── config.py       # YAML/Json config parsers
    ├── constants.py    # Global hardcoded values
    ├── device.py       # Hardware auto-detection & management
    ├── enum.py         # Global Enums
    ├── factory.py      # Register & Factory
    ├── fileio.py       # Atomic and optimized read/write operations
    ├── filesystem.py   # Filesystem operations
    ├── logger.py       # Unified terminal (Rich) and file-based logging
    ├── path.py         # Path manipulation
    ├── patterns.py     # Design patterns (Singleton, Factory, etc.)
    ├── profile.py      # Benchmark, FLOPs, MACs
    ├── singleton.py    # Singleton pattern
    ├── system.py       # Seeding, environment auditing, and process control
    ├── typing.py       # Type hints
    ├── ui.py           # Terminal & GUI
    └── utils.py        # General purpose helpers
"""

from __future__ import annotations

from .config import *
from .constants import *
from .data import *
# from .device import *
from .enum import *
from .factory import *
from .fileio import *
from .filesystem import *
from .logger import *
from .path import *
from .profile import *
from .singleton import *
from .system import *
from .typing import *
from .ui import *
from .utils import *
