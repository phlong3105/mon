#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Base Architectures.

This package includes base meta-architectures used across domains.

File Structure:
::

    models/
    ├── __init__.py
    ├── backbone/       # Feature extractors
    ├── head/           # Output heads
    └── neck/           # Feature aggregators
"""

from __future__ import annotations

from .backbone import *
from .head import *
from .neck import *
