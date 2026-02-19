#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Data Structures.

This package contains the containers that carry information throughout the
``mon`` package.

File Structure:
::

    data/
    ├── __init__.py
    ├── bbox.py         #
    ├── class_def.py    #
    ├── data.py         #
    ├── device.py       # Hardware auto-detection & management
    ├── image.py        #
    ├── mask.py         #
    ├── prob.py         #
    ├── timer.py        #
    └── weights.py      # General purpose helpers
"""

from __future__ import annotations

from .bbox import *
from .class_def import *
from .data import *
from .device import *
from .image import *
from .mask import *
from .prob import *
from .timer import *
from .weights import *
