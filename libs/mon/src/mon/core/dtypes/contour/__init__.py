#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Contour data type.

This package contains a "full-stack" toolkit for contour data, including data
structure, ingestion, analysis, atomic transformations, complex workflows, and
rendering utilities.
"""

__all__ = [
    "convert",
    "denormalize",
    "normalize",
]

from .core import *
from .io import *
from .meta import *
from .ops import *
from .proc import *
from .vis import *

