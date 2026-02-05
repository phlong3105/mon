#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Image enhancement algorithms.

This package contains various image enhancement methods commonly used in
computer vision.

Notes:
    - Design Pattern: Multiple Component-Based Frameworks.
    - Goal: Encapsulate multiple "Component-Based Frameworks" for image
      enhancement.
    - Structure:
        ::

            enhance/
            ├── __init__.py             # Unified entry point
            ├──lle/
            │   ├── __init__.py         # Exposes all
            │   ├── base.py             # Base classes and mixins
            │   ├── comp/               # Components
            │   ├── impl/               # Implementations
            │   ├── adapters/           # Adapters
            │   └── utils.py            # Utilities and helpers
            └── ...
"""

from __future__ import annotations

from .lle import *
