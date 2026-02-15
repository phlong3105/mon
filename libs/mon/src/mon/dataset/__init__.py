#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Dataset.

This package contains "full-stack" dataset utilities and several concrete
dataset implementations.

File Structure:
::

    mon.core/
    ├── __init__.py
    ├── augment/        # Augmentation pipelines
    ├── base/           # Base components (datasets, dataloaders, etc.)
    ├── transforms/     # Data transforms
    ├── zoo/            # Concrete dataset implementations
    └── utils.py        # General purpose helpers
"""

from __future__ import annotations

from . import transforms as T
from .base import *
from .transforms import build_compose
from .utils import *
from .zoo import *
