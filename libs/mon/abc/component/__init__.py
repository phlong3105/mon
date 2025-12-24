#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""<Name>.

This package contains various <Name>s commonly used in <application domain>.

This is the Component-Based Framework. It is useful to organize a family of
related functionalities that share a common interface/inheritance but aren't tie
to the specific "interchanged algorithm" requirement of the Strategy Pattern.

The Component-Based Framework typically has the following structure:
    component/
    ├── __init__.py    # Exposes all concrete classes
    ├── base.py        # The Abstract Base Class (ABC)
    ├── ...
    └── utils.py       # Utility functions and helpers
"""

__all__ = []

from .base import *
from .utils import *


# ==============================================================================
# REGISTRY & FACTORY (Type Resolution)
# ==============================================================================

# --- Register (Adding new spokes to the hub) ---


# --- Resolve (Retrieving spokes by name/key) ---
