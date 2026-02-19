#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Base and Generic.

This package contains the base classes and generic data structures.

File Structure:
::

    mon.core/
    ├── __init__.py
    ├── collection.py   # Generic collection data structures
    ├── decorator.py    # Generic decorators
    └── enum.py         # Custom Enum classes
"""

from __future__ import annotations

from .collection import *
from .decorator import *
from .enum import *
