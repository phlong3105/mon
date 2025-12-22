#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Depth data type.

This package contains a "full-stack" toolkit for depth data, including data
structure, ingestion, analysis, atomic transformations, complex workflows, and
rendering utilities.
"""

__all__ = [
    "DepthMap",
    "to_color",
]

from .core import *
from .io import *
from .meta import *
from .ops import *
from .proc import *
from .vis import *
