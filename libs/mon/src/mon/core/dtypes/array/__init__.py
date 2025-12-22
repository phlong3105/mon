#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Array-like data type.

This package contains a "full-stack" toolkit for array-like data, including
data structure, ingestion, analysis, atomic transformations, complex workflows,
and rendering utilities.
"""

__all__ = [
    "TensorOrArray",
]

from .core import *
from .io import *
from .meta import *
from .ops import *
from .proc import *
from .vis import *

