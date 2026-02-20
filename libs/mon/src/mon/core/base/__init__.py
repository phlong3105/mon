#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Base and Generic.

This package contains the base classes and generic data structures.

File Structure:
::

    mon.core/
    ├── __init__.py
    ├── collection.py   # Generic collection data structures
    ├── enum.py         # Custom Enum classes
    └── singleton.py    # Singleton mechanism
"""

from __future__ import annotations

from .collection import *
from .enum import *
from .singleton import *
