#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Mask data type.

This package contains a "full-stack" toolkit for mask data, including data
structure, ingestion, analysis, atomic transformations, complex workflows, and
rendering utilities.
"""

__all__ = [
    "SemanticMask",
]

from .core import *
from .io import *
from .meta import *
from .ops import *
from .proc import *
from .vis import *

