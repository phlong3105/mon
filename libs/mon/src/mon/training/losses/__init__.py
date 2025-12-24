#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Loss functions.

This package contains various loss functions commonly used in training machine
learning models, particularly in computer vision tasks.

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

from .base import *
from .basic import *
from .external import *
from .image import *
from .utils import *

# ==============================================================================
# REGISTRY & FACTORY (Type Resolution)
# ==============================================================================

# --- Register (Adding new spokes to the hub) ---


# --- Resolve (Retrieving spokes by name/key) ---
