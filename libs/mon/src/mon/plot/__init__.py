#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Plotting.

This package contains plotting utilities. The purpose of this package is to
define a consistent interface and configuration for all plotting operations.

File Structure:
::

    plot/
    ├── __init__.py
    ├── base.py         # Base plotting functionality.
    ├── bar.py          # Bar chart plotting functionality.
    ├── line.py         # Line chart plotting functionality.
    └── polar.py        # Polar chart plotting functionality.
"""

from __future__ import annotations

from .bar import *
from .base import *
from .line import *
from .polar import *
