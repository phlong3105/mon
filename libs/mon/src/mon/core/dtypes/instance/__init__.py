#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Instance annotation.

This package contains a "full-stack" toolkit for instance annotation, including
data structure, ingestion, analysis, atomic transformations, complex workflows,
and rendering utilities.
"""

__all__ = [
    "Instance",
]

from .core import *
from .io import *
from .meta import *
from .ops import *
from .proc import *
from .vis import *
