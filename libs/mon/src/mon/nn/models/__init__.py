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
    ├── neck/           # Feature aggregators
    ├── repr/           # Representation learning
    ├── upsample/       # Feature upsampling methods
    └── base.py         # Base model and mixins
"""

from __future__ import annotations

from .backbone import *
from .base import *
from .head import *
from .neck import *
from .repr import *
from .upsample import *
